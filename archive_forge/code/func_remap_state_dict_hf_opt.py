import math
import re
from collections import OrderedDict
import torch
import torch.nn.functional as F
from transformers import GPT2Config, OPTConfig
def remap_state_dict_hf_opt(state_dict, config):

    def key_mapping_model(key):
        key = re.sub('^model.decoder.', 'transformer.', key)
        key = re.sub('^decoder.', 'transformer.', key)
        return key
    state_dict = OrderedDict(((key_mapping_model(k), v) for k, v in state_dict.items()))

    def key_mapping_emb(key):
        key = re.sub('^transformer.embed_tokens.', 'transformer.embeddings.word_embeddings.', key)
        key = re.sub('^transformer.project_in.', 'transformer.embeddings.project_in.', key)
        key = re.sub('^transformer.project_out.', 'project_out.', key)
        key = re.sub('^transformer.embed_positions.', 'transformer.embeddings.position_embeddings.', key)
        return key
    state_dict = OrderedDict(((key_mapping_emb(k), v) for k, v in state_dict.items()))
    pos_embeddings = state_dict.pop('transformer.embeddings.position_embeddings.weight')
    state_dict['transformer.embeddings.position_embeddings.weight'] = pos_embeddings[2:]
    word_embeddings = state_dict.pop('transformer.embeddings.word_embeddings.weight')
    pad_vocab_size_multiple = getattr(config, 'pad_vocab_size_multiple', 1)
    vocab_size = math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple
    state_dict['transformer.embeddings.word_embeddings.weight'] = F.pad(word_embeddings, (0, 0, 0, vocab_size - word_embeddings.shape[0]))
    state_dict['lm_head.weight'] = state_dict['transformer.embeddings.word_embeddings.weight']

    def key_mapping_ln(key):
        key = re.sub('^transformer.final_layer_norm.', 'transformer.ln_f.', key)
        key = re.sub('^transformer.layer_norm.', 'transformer.ln_f.', key)
        key = re.sub('^transformer.layers.(\\d+).self_attn_layer_norm.', 'transformer.layers.\\1.norm1.', key)
        key = re.sub('^transformer.layers.(\\d+).final_layer_norm.', 'transformer.layers.\\1.norm2.', key)
        return key
    state_dict = OrderedDict(((key_mapping_ln(k), v) for k, v in state_dict.items()))

    def key_mapping_mlp(key):
        return re.sub('^transformer.layers.(\\d+).fc(1|2).', 'transformer.layers.\\1.mlp.fc\\2.', key)
    state_dict = OrderedDict(((key_mapping_mlp(k), v) for k, v in state_dict.items()))
    for l in range(config.n_layer):
        Wq = state_dict.pop(f'transformer.layers.{l}.self_attn.q_proj.weight')
        Wk = state_dict.pop(f'transformer.layers.{l}.self_attn.k_proj.weight')
        Wv = state_dict.pop(f'transformer.layers.{l}.self_attn.v_proj.weight')
        bq = state_dict.pop(f'transformer.layers.{l}.self_attn.q_proj.bias')
        bk = state_dict.pop(f'transformer.layers.{l}.self_attn.k_proj.bias')
        bv = state_dict.pop(f'transformer.layers.{l}.self_attn.v_proj.bias')
        state_dict[f'transformer.layers.{l}.mixer.Wqkv.weight'] = torch.cat([Wq, Wk, Wv], dim=0)
        state_dict[f'transformer.layers.{l}.mixer.Wqkv.bias'] = torch.cat([bq, bk, bv], dim=0)

    def key_mapping_attn(key):
        return re.sub('^transformer.layers.(\\d+).self_attn.out_proj.', 'transformer.layers.\\1.mixer.out_proj.', key)
    state_dict = OrderedDict(((key_mapping_attn(k), v) for k, v in state_dict.items()))
    return state_dict