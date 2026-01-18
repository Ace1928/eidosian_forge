from collections import OrderedDict
import os
import torch
from torch.serialization import default_restore_location
from typing import Any, Dict, List
from parlai.core.agents import create_agent
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript
def convert_model_weight(self, opt: Opt) -> Dict[str, Any]:
    """
        Convert state_dict between fairseq and ParlAI.

        :param opt:
            ParlAI opt

        :return state_dict:
            return a state dict to load into ParlAI model.
        """
    state = self.state
    agent = self.agent
    state_dict = state['model']
    return_dict = OrderedDict()
    for each_key in state_dict.keys():
        mapped_key = each_key
        if mapped_key == 'encoder.version' or mapped_key == 'decoder.version':
            continue
        for emb in EMBEDDING_DICT_MAPPING:
            mapped_key = mapped_key.replace(emb, EMBEDDING_DICT_MAPPING[emb])
        if 'encoder' in each_key and 'self_attn' in each_key:
            mapped_key = mapped_key.replace('self_attn', 'attention')
        elif 'decoder' in each_key and 'self_attn' in each_key:
            mapped_key = mapped_key.replace('self_attn', 'self_attention')
        elif 'decoder' in each_key and 'encoder_attn' in each_key:
            mapped_key = mapped_key.replace('encoder_attn', 'encoder_attention')
        if 'in_proj_weight' in mapped_key or 'in_proj_bias' in mapped_key:
            for weightorbias in {'weight', 'bias'}:
                attention_project_name = 'in_proj_{}'.format(weightorbias)
                if attention_project_name in mapped_key:
                    weight = state_dict[each_key]
                    size = int(weight.size(0) / 3)
                    weights = weight.split(size, 0)
                    return_dict[mapped_key.replace(attention_project_name, 'q_lin.{}'.format(weightorbias))] = weights[0]
                    return_dict[mapped_key.replace(attention_project_name, 'k_lin.{}'.format(weightorbias))] = weights[1]
                    return_dict[mapped_key.replace(attention_project_name, 'v_lin.{}'.format(weightorbias))] = weights[2]
            continue
        elif 'v_proj' in mapped_key or 'k_proj' in mapped_key or 'q_proj' in mapped_key:
            mapped_key = mapped_key.replace('v_proj', 'v_lin')
            mapped_key = mapped_key.replace('q_proj', 'q_lin')
            mapped_key = mapped_key.replace('k_proj', 'k_lin')
        for old, new in FFN_MAPPING.items():
            mapped_key = mapped_key.replace(old, new)
        if 'encoder.' in mapped_key:
            mapped_key = mapped_key.replace('attention_layer_norm', 'norm1')
            mapped_key = mapped_key.replace('final_layer_norm', 'norm2')
        else:
            mapped_key = mapped_key.replace('self_attention_layer_norm', 'norm1')
            mapped_key = mapped_key.replace('encoder_attention_layer_norm', 'norm2')
            mapped_key = mapped_key.replace('final_layer_norm', 'norm3')
        for _key in ['encoder', 'decoder']:
            mapped_key = mapped_key.replace(f'{_key}.layer_norm', f'{_key}.norm_embeddings')
            mapped_key = mapped_key.replace(f'{_key}.layernorm_embedding', f'{_key}.norm_embeddings')
        weight = state_dict[each_key]
        return_dict[mapped_key] = weight
    enc_emb_key = 'encoder.embeddings.weight'
    bart_dict = os.path.join(opt['datapath'], 'models/bart/bart.large/dict.txt')
    with open(bart_dict) as f:
        offset_dict = {i: l.split()[0] for i, l in enumerate(f.readlines())}
    new_embs = return_dict[enc_emb_key].clone()
    for idx, new_idx in offset_dict.items():
        try:
            new_embs[int(new_idx) + 4] = return_dict[enc_emb_key][idx + 4]
        except ValueError:
            if 'madeupword' in new_idx:
                pad_idx = int(new_idx.split('madeupword')[1])
                new_embs[-(4 - pad_idx)] = return_dict['encoder.embeddings.weight'][idx + 4]
    return_dict['encoder.embeddings.weight'] = new_embs
    size_dict = return_dict[enc_emb_key].size(0)
    if size_dict == len(agent.dict) + 1 and '<mask>' not in agent.dict:
        return_dict[enc_emb_key] = return_dict[enc_emb_key][:size_dict - 1, :]
        size_dict -= 1
    specials, words = return_dict[enc_emb_key].split([4, size_dict - 4], 0)
    bos, pad, eos, unk = specials
    if not self.opt['retain_bos_emb']:
        bos = eos
    specials = torch.stack([pad, bos, eos, unk])
    fp16_pad = (8 - (len(specials) + len(words)) % 8) % 8
    fp16_pad_ez = torch.zeros(fp16_pad, specials.size(1)).type_as(specials)
    return_dict[enc_emb_key] = torch.cat([specials, words, fp16_pad_ez], 0)
    return_dict['decoder.embeddings.weight'] = return_dict[enc_emb_key]
    return_dict['embeddings.weight'] = return_dict[enc_emb_key]
    if 'encoder.position_embeddings.weight' in return_dict:
        return_dict['encoder.position_embeddings.weight'] = return_dict['encoder.position_embeddings.weight'][2:, :]
        return_dict['decoder.position_embeddings.weight'] = return_dict['decoder.position_embeddings.weight'][2:, :]
    else:
        from fairseq.modules.sinusoidal_positional_embedding import SinusoidalPositionalEmbedding
        emb = SinusoidalPositionalEmbedding.get_embedding(128 + 2, opt['embedding_size'], 1)
        del return_dict['encoder.position_embeddings._float_tensor']
        del return_dict['decoder.position_embeddings._float_tensor']
        return_dict['encoder.position_embeddings.weight'] = emb[2:]
        return_dict['decoder.position_embeddings.weight'] = emb[2:]
    return_dict['START'] = torch.LongTensor([1])
    return return_dict