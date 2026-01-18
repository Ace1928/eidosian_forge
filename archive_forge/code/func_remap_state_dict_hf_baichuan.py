import math
import json
import re
from pathlib import Path
from collections import OrderedDict
import torch
import torch.nn.functional as F
from einops import rearrange
from transformers import GPT2Config, AutoConfig, PretrainedConfig
def remap_state_dict_hf_baichuan(state_dict, config):

    def key_mapping_layers(key):
        return re.sub('^model.', 'transformer.', key)
    state_dict = OrderedDict(((key_mapping_layers(k), v) for k, v in state_dict.items()))

    def key_mapping_emb(key):
        return re.sub('^transformer.embed_tokens.', 'transformer.embeddings.word_embeddings.', key)
    state_dict = OrderedDict(((key_mapping_emb(k), v) for k, v in state_dict.items()))
    word_embeddings = state_dict.pop('transformer.embeddings.word_embeddings.weight')
    pad_vocab_size_multiple = getattr(config, 'pad_vocab_size_multiple', 1)
    vocab_size = math.ceil(word_embeddings.shape[0] / pad_vocab_size_multiple) * pad_vocab_size_multiple
    state_dict['transformer.embeddings.word_embeddings.weight'] = F.pad(word_embeddings, (0, 0, 0, vocab_size - word_embeddings.shape[0]))
    if getattr(config, 'tie_word_embeddings'):
        state_dict['lm_head.weight'] = state_dict['transformer.embeddings.word_embeddings.weight']
    else:
        output_embeddings = state_dict.pop('lm_head.weight')
        vocab_size = math.ceil(output_embeddings.shape[0] / pad_vocab_size_multiple) * pad_vocab_size_multiple
        state_dict['lm_head.weight'] = F.pad(output_embeddings, (0, 0, 0, vocab_size - output_embeddings.shape[0]))

    def key_mapping_ln(key):
        key = re.sub('^transformer.norm.', 'transformer.ln_f.', key)
        key = re.sub('^transformer.layers.(\\d+).input_layernorm.', 'transformer.layers.\\1.norm1.', key)
        key = re.sub('^transformer.layers.(\\d+).post_attention_layernorm.', 'transformer.layers.\\1.norm2.', key)
        return key
    state_dict = OrderedDict(((key_mapping_ln(k), v) for k, v in state_dict.items()))
    for l in range(config.n_layer):
        w1 = state_dict.pop(f'transformer.layers.{l}.mlp.gate_proj.weight')
        w3 = state_dict.pop(f'transformer.layers.{l}.mlp.up_proj.weight')
        state_dict[f'transformer.layers.{l}.mlp.fc1.weight'] = torch.cat([w3, w1], dim=0)

    def key_mapping_mlp(key):
        return re.sub('^transformer.layers.(\\d+).mlp.down_proj.', 'transformer.layers.\\1.mlp.fc2.', key)
    state_dict = OrderedDict(((key_mapping_mlp(k), v) for k, v in state_dict.items()))

    def key_mapping_attn(key):
        key = re.sub('^transformer.layers.(\\d+).self_attn.W_pack.', 'transformer.layers.\\1.mixer.Wqkv.', key)
        key = re.sub('^transformer.layers.(\\d+).self_attn.o_proj.', 'transformer.layers.\\1.mixer.out_proj.', key)
        return key
    state_dict = OrderedDict(((key_mapping_attn(k), v) for k, v in state_dict.items()))
    for l in range(config.n_layer):
        state_dict.pop(f'transformer.layers.{l}.self_attn.rotary_emb.inv_freq', None)
    return state_dict