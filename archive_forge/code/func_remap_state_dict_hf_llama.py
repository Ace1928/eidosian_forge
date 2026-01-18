import json
import math
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Union
import torch
import torch.nn.functional as F
from sentencepiece import SentencePieceProcessor
from transformers import GPT2Config, LlamaConfig
from einops import rearrange
def remap_state_dict_hf_llama(state_dict: Dict[str, torch.Tensor], config: GPT2Config) -> Dict[str, torch.Tensor]:
    """Convert the state_dict in Hugging Face format to standard GPT format.

    This function modifies state_dict in place.
    """

    def key_mapping_emb(key):
        return re.sub('^model.embed_tokens.', 'transformer.embeddings.word_embeddings.', key)
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
    for l in range(config.n_layer):
        w1 = state_dict.pop(f'model.layers.{l}.mlp.gate_proj.weight')
        w3 = state_dict.pop(f'model.layers.{l}.mlp.up_proj.weight')
        state_dict[f'transformer.layers.{l}.mlp.fc1.weight'] = torch.cat([w3, w1], dim=0)

    def key_mapping_mlp(key):
        return re.sub('^model.layers.(\\d+).mlp.down_proj.', 'transformer.layers.\\1.mlp.fc2.', key)
    state_dict = OrderedDict(((key_mapping_mlp(k), v) for k, v in state_dict.items()))

    def key_mapping_ln(key):
        key = re.sub('^model.norm.', 'transformer.ln_f.', key)
        key = re.sub('^model.layers.(\\d+).input_layernorm.', 'transformer.layers.\\1.norm1.', key)
        key = re.sub('^model.layers.(\\d+).post_attention_layernorm.', 'transformer.layers.\\1.norm2.', key)
        return key
    state_dict = OrderedDict(((key_mapping_ln(k), v) for k, v in state_dict.items()))

    def inv_permute(w):
        return rearrange(w, '(h two d) n -> (h d two) n', d=config.n_embd // config.n_head // 2, two=2)
    for l in range(config.n_layer):
        Wq = state_dict.pop(f'model.layers.{l}.self_attn.q_proj.weight')
        Wk = state_dict.pop(f'model.layers.{l}.self_attn.k_proj.weight')
        Wv = state_dict.pop(f'model.layers.{l}.self_attn.v_proj.weight')
        state_dict[f'transformer.layers.{l}.mixer.Wqkv.weight'] = torch.cat([inv_permute(Wq), inv_permute(Wk), Wv], dim=0)
        state_dict.pop(f'model.layers.{l}.self_attn.rotary_emb.inv_freq', None)

    def key_mapping_attn(key):
        return re.sub('^model.layers.(\\d+).self_attn.o_proj.', 'transformer.layers.\\1.mixer.out_proj.', key)
    state_dict = OrderedDict(((key_mapping_attn(k), v) for k, v in state_dict.items()))
    return state_dict