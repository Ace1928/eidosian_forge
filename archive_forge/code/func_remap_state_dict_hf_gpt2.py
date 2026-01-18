import logging
import math
import re
from collections import OrderedDict, namedtuple
from collections.abc import Sequence
from functools import partial
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import GPT2Config
from flash_attn.models.bigcode import remap_state_dict_hf_bigcode
from flash_attn.models.falcon import remap_state_dict_hf_falcon
from flash_attn.models.gpt_neox import remap_state_dict_hf_gpt_neox
from flash_attn.models.gptj import remap_state_dict_hf_gptj
from flash_attn.models.llama import remap_state_dict_hf_llama
from flash_attn.models.opt import remap_state_dict_hf_opt
from flash_attn.modules.block import Block, ParallelBlock
from flash_attn.modules.embedding import GPT2Embeddings, ParallelGPT2Embeddings
from flash_attn.modules.mha import MHA, ParallelMHA
from flash_attn.modules.mlp import (
from flash_attn.ops.activations import sqrelu_fwd
from flash_attn.utils.distributed import (
from flash_attn.utils.generation import GenerationMixin
from flash_attn.utils.pretrained import state_dict_from_pretrained
def remap_state_dict_hf_gpt2(state_dict, config):

    def key_mapping_pos_emb(key):
        return re.sub('^wpe.', 'transformer.embeddings.position_embeddings.', key)
    state_dict = OrderedDict(((key_mapping_pos_emb(k), v) for k, v in state_dict.items()))
    word_embeddings = state_dict.pop('wte.weight')
    pad_vocab_size_multiple = getattr(config, 'pad_vocab_size_multiple', 1)
    vocab_size = math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple
    state_dict['transformer.embeddings.word_embeddings.weight'] = F.pad(word_embeddings, (0, 0, 0, vocab_size - word_embeddings.shape[0]))
    state_dict['lm_head.weight'] = state_dict['transformer.embeddings.word_embeddings.weight']

    def key_mapping_ln(key):
        key = re.sub('^ln_f.(weight|bias)', 'transformer.ln_f.\\1', key)
        key = re.sub('^h.(\\d+).ln_(1|2).(weight|bias)', 'transformer.layers.\\1.norm\\2.\\3', key)
        return key
    state_dict = OrderedDict(((key_mapping_ln(k), v) for k, v in state_dict.items()))
    for d in range(config.num_hidden_layers):
        W1 = state_dict.pop(f'h.{d}.mlp.c_fc.weight')
        state_dict[f'transformer.layers.{d}.mlp.fc1.weight'] = W1.t()
        W2 = state_dict.pop(f'h.{d}.mlp.c_proj.weight')
        state_dict[f'transformer.layers.{d}.mlp.fc2.weight'] = W2.t()

    def key_mapping_mlp(key):
        key = re.sub('^h.(\\d+).mlp.c_fc.bias', 'transformer.layers.\\1.mlp.fc1.bias', key)
        key = re.sub('^h.(\\d+).mlp.c_proj.bias', 'transformer.layers.\\1.mlp.fc2.bias', key)
        return key
    state_dict = OrderedDict(((key_mapping_mlp(k), v) for k, v in state_dict.items()))
    for d in range(config.num_hidden_layers):
        state_dict.pop(f'h.{d}.attn.bias')
        Wqkv = state_dict.pop(f'h.{d}.attn.c_attn.weight')
        state_dict[f'transformer.layers.{d}.mixer.Wqkv.weight'] = Wqkv.t()
        Wout = state_dict.pop(f'h.{d}.attn.c_proj.weight')
        state_dict[f'transformer.layers.{d}.mixer.out_proj.weight'] = Wout.t()

    def key_mapping_attn(key):
        key = re.sub('^h.(\\d+).attn.c_attn.bias', 'transformer.layers.\\1.mixer.Wqkv.bias', key)
        key = re.sub('^h.(\\d+).attn.c_proj.bias', 'transformer.layers.\\1.mixer.out_proj.bias', key)
        return key
    state_dict = OrderedDict(((key_mapping_attn(k), v) for k, v in state_dict.items()))
    return state_dict