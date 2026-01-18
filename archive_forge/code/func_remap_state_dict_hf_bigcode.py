import math
import re
from collections import OrderedDict
import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPTBigCodeConfig, PretrainedConfig
def remap_state_dict_hf_bigcode(state_dict, config: PretrainedConfig):
    """
    Map the state_dict of a Huggingface BigCode model to be flash_attn compatible.
    """

    def key_mapping_pos_emb(key):
        return re.sub('^transformer.wpe.', 'transformer.embeddings.position_embeddings.', key)
    state_dict = OrderedDict(((key_mapping_pos_emb(k), v) for k, v in state_dict.items()))
    word_embeddings = state_dict.pop('transformer.wte.weight')
    pad_vocab_size_multiple = getattr(config, 'pad_vocab_size_multiple', 1)
    vocab_size = math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple
    state_dict['transformer.embeddings.word_embeddings.weight'] = F.pad(word_embeddings, (0, 0, 0, vocab_size - word_embeddings.shape[0]))
    state_dict['lm_head.weight'] = state_dict['transformer.embeddings.word_embeddings.weight']

    def key_mapping_ln(key):
        key = re.sub('^transformer.ln_f.(weight|bias)', 'transformer.ln_f.\\1', key)
        key = re.sub('^transformer.h.(\\d+).ln_(1|2).(weight|bias)', 'transformer.layers.\\1.norm\\2.\\3', key)
        return key
    state_dict = OrderedDict(((key_mapping_ln(k), v) for k, v in state_dict.items()))

    def key_mapping_mlp(key):
        key = re.sub('^transformer.h.(\\d+).mlp.c_fc.weight', 'transformer.layers.\\1.mlp.fc1.weight', key)
        key = re.sub('^transformer.h.(\\d+).mlp.c_proj.weight', 'transformer.layers.\\1.mlp.fc2.weight', key)
        key = re.sub('^transformer.h.(\\d+).mlp.c_fc.bias', 'transformer.layers.\\1.mlp.fc1.bias', key)
        key = re.sub('^transformer.h.(\\d+).mlp.c_proj.bias', 'transformer.layers.\\1.mlp.fc2.bias', key)
        return key
    state_dict = OrderedDict(((key_mapping_mlp(k), v) for k, v in state_dict.items()))
    assert config.multi_query, 'Only multi-query attention is supported'
    for d in range(config.num_hidden_layers):
        embed_dim = config.n_embd
        head_dim = embed_dim // config.n_head
        c_attn_weight = state_dict.pop(f'transformer.h.{d}.attn.c_attn.weight')
        q, k, v = torch.split(c_attn_weight, [embed_dim, head_dim, head_dim], dim=0)
        k = torch.tile(k, (config.n_head, 1))
        v = torch.tile(v, (config.n_head, 1))
        state_dict[f'transformer.layers.{d}.mixer.Wqkv.weight'] = torch.cat((q, k, v), dim=0)
        c_attn_bias = state_dict.pop(f'transformer.h.{d}.attn.c_attn.bias')
        q, k, v = torch.split(c_attn_bias, [embed_dim, head_dim, head_dim], dim=0)
        k = torch.tile(k, (config.n_head,))
        v = torch.tile(v, (config.n_head,))
        state_dict[f'transformer.layers.{d}.mixer.Wqkv.bias'] = torch.cat((q, k, v), dim=0)

    def key_mapping_attn(key):
        key = re.sub('^transformer.h.(\\d+).attn.c_proj.weight', 'transformer.layers.\\1.mixer.out_proj.weight', key)
        key = re.sub('^transformer.h.(\\d+).attn.c_proj.bias', 'transformer.layers.\\1.mixer.out_proj.bias', key)
        return key
    state_dict = OrderedDict(((key_mapping_attn(k), v) for k, v in state_dict.items()))
    return state_dict