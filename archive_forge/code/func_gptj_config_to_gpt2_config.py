import math
import re
from collections import OrderedDict
import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPTJConfig
def gptj_config_to_gpt2_config(gptj_config: GPTJConfig) -> GPT2Config:
    headdim = gptj_config.n_embd // gptj_config.n_head
    return GPT2Config(vocab_size=gptj_config.vocab_size, n_positions=0, n_embd=gptj_config.n_embd, n_layer=gptj_config.n_layer, n_head=gptj_config.n_head, n_inner=gptj_config.n_inner, activation_function=gptj_config.activation_function, resid_pdrop=gptj_config.resid_pdrop, embd_pdrop=gptj_config.embd_pdrop, attn_pdrop=gptj_config.attn_pdrop, layer_norm_epsilon=gptj_config.layer_norm_epsilon, initializer_range=gptj_config.initializer_range, bos_token_id=gptj_config.bos_token_id, eos_token_id=gptj_config.eos_token_id, prenorm=True, parallel_block=True, parallel_block_tied_norm=True, rotary_emb_fraction=gptj_config.rotary_dim / headdim, rotary_emb_interleaved=True, tie_word_embeddings=False, qkv_proj_bias=False, out_proj_bias=False, lm_head_bias=True)