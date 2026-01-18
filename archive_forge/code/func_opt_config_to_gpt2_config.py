import math
import re
from collections import OrderedDict
import torch
import torch.nn.functional as F
from transformers import GPT2Config, OPTConfig
def opt_config_to_gpt2_config(opt_config: OPTConfig) -> GPT2Config:
    assert opt_config.layerdrop == 0.0
    assert opt_config.layer_norm_elementwise_affine
    word_embed_proj_dim = None if opt_config.word_embed_proj_dim == opt_config.hidden_size else opt_config.word_embed_proj_dim
    return GPT2Config(vocab_size=opt_config.vocab_size, n_positions=opt_config.max_position_embeddings, n_embd=opt_config.hidden_size, n_layer=opt_config.num_hidden_layers, n_head=opt_config.num_attention_heads, n_inner=opt_config.ffn_dim, activation_function=opt_config.activation_function, resid_pdrop=opt_config.dropout, embd_pdrop=opt_config.dropout, attn_pdrop=opt_config.attention_dropout, initializer_range=opt_config.init_std, bos_token_id=opt_config.bos_token_id, eos_token_id=opt_config.eos_token_id, prenorm=opt_config.do_layer_norm_before, word_embed_proj_dim=word_embed_proj_dim)