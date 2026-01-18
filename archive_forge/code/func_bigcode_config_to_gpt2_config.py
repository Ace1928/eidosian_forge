import math
import re
from collections import OrderedDict
import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPTBigCodeConfig, PretrainedConfig
def bigcode_config_to_gpt2_config(bigcode_config: GPTBigCodeConfig) -> GPT2Config:
    return GPT2Config(activation_function=bigcode_config.activation_function, attn_pdrop=bigcode_config.attn_pdrop, bos_token_id=bigcode_config.bos_token_id, embd_pdrop=bigcode_config.embd_pdrop, eos_token_id=bigcode_config.eos_token_id, initializer_range=bigcode_config.initializer_range, layer_norm_epsilon=bigcode_config.layer_norm_epsilon, max_batch_size=bigcode_config.max_batch_size, max_sequence_length=bigcode_config.max_sequence_length, model_type=bigcode_config.model_type, multi_query=bigcode_config.multi_query, n_embd=bigcode_config.n_embd, n_head=bigcode_config.n_head, n_inner=bigcode_config.n_inner, n_layer=bigcode_config.n_layer, n_positions=bigcode_config.n_positions, resid_pdrop=bigcode_config.resid_pdrop, scale_attn_weights=bigcode_config.scale_attn_weights, summary_activation=bigcode_config.summary_activation, summary_first_dropout=bigcode_config.summary_first_dropout, summary_proj_to_labels=bigcode_config.summary_proj_to_labels, summary_type=bigcode_config.summary_type, summary_use_proj=bigcode_config.summary_use_proj, use_cache=bigcode_config.use_cache, vocab_size=bigcode_config.vocab_size)