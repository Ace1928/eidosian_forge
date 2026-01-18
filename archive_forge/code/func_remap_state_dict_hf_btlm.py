import math
import json
import re
from pathlib import Path
from collections import OrderedDict
import torch
import torch.nn.functional as F
from einops import rearrange
from transformers import GPT2Config, AutoConfig, PretrainedConfig
def remap_state_dict_hf_btlm(state_dict, config):

    def key_mapping_pos_emb(key):
        return re.sub('^transformer.wpe.', 'transformer.embeddings.position_embeddings.', key)
    if 'transformer.wpe.weight' in state_dict:
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
    for d in range(config.num_hidden_layers):
        W1 = state_dict.pop(f'transformer.h.{d}.mlp.c_fc.weight')
        W3 = state_dict.pop(f'transformer.h.{d}.mlp.c_fc2.weight')
        state_dict[f'transformer.layers.{d}.mlp.fc1.weight'] = torch.cat([W1.t(), W3.t()], dim=0)
        b1 = state_dict.pop(f'transformer.h.{d}.mlp.c_fc.bias')
        b3 = state_dict.pop(f'transformer.h.{d}.mlp.c_fc2.bias')
        state_dict[f'transformer.layers.{d}.mlp.fc1.bias'] = torch.cat([b1, b3], dim=0)
        W2 = state_dict.pop(f'transformer.h.{d}.mlp.c_proj.weight')
        state_dict[f'transformer.layers.{d}.mlp.fc2.weight'] = W2.t()

    def key_mapping_mlp(key):
        key = re.sub('^transformer.h.(\\d+).mlp.c_proj.bias', 'transformer.layers.\\1.mlp.fc2.bias', key)
        return key
    state_dict = OrderedDict(((key_mapping_mlp(k), v) for k, v in state_dict.items()))
    for d in range(config.num_hidden_layers):
        Wqkv = state_dict.pop(f'transformer.h.{d}.attn.c_attn.weight')
        state_dict[f'transformer.layers.{d}.mixer.Wqkv.weight'] = Wqkv.t()
        Wout = state_dict.pop(f'transformer.h.{d}.attn.c_proj.weight')
        state_dict[f'transformer.layers.{d}.mixer.out_proj.weight'] = Wout.t()
    state_dict.pop(f'transformer.relative_pe.slopes')

    def key_mapping_attn(key):
        key = re.sub('^transformer.h.(\\d+).attn.c_attn.bias', 'transformer.layers.\\1.mixer.Wqkv.bias', key)
        key = re.sub('^transformer.h.(\\d+).attn.c_proj.bias', 'transformer.layers.\\1.mixer.out_proj.bias', key)
        return key
    state_dict = OrderedDict(((key_mapping_attn(k), v) for k, v in state_dict.items()))
    return state_dict