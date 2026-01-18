import math
import json
import re
from pathlib import Path
from collections import OrderedDict
import torch
import torch.nn.functional as F
from einops import rearrange
from transformers import GPT2Config, AutoConfig, PretrainedConfig
def btlm_config_to_gpt2_config(btlm_config: PretrainedConfig) -> GPT2Config:
    return GPT2Config(vocab_size=btlm_config.vocab_size, n_positions=0 if btlm_config.position_embedding_type == 'alibi' else btlm_config.n_positions, n_embd=btlm_config.hidden_size, n_layer=btlm_config.num_hidden_layers, n_head=btlm_config.num_attention_heads, n_inner=btlm_config.n_inner, activation_function=btlm_config.activation_function, resid_pdrop=btlm_config.resid_pdrop, embd_pdrop=btlm_config.embd_pdrop, attn_pdrop=btlm_config.attn_pdrop, layer_norm_epsilon=btlm_config.layer_norm_epsilon, initializer_range=btlm_config.initializer_range, bos_token_id=btlm_config.bos_token_id, eos_token_id=btlm_config.eos_token_id, use_alibi=btlm_config.position_embedding_type == 'alibi', use_flash_attn=btlm_config.position_embedding_type == 'alibi', mup_width_scale=btlm_config.mup_width_scale, mup_embeddings_multiplier=btlm_config.mup_embeddings_scale, mup_output_multiplier=btlm_config.mup_output_alpha, mup_scale_qk_dot_by_d=btlm_config.mup_scale_qk_dot_by_d, mlp_multiple_of=1)