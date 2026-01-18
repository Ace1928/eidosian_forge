import argparse
import copy
import enum
import functools
import os
import typing
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, get_args
import torch
from .constants import FSDP_AUTO_WRAP_POLICY, FSDP_BACKWARD_PREFETCH, FSDP_SHARDING_STRATEGY, FSDP_STATE_DICT_TYPE
from .environment import str_to_bool
from .imports import is_cuda_available, is_npu_available, is_xpu_available
from .versions import compare_versions
def set_network_size_args(self, model, batch_data=None):
    if 'megatron-bert' in model.config.model_type.lower():
        model_type_name = 'bert'
        num_layers = model.config.num_hidden_layers
        hidden_size = model.config.hidden_size
        num_attention_heads = model.config.num_attention_heads
        max_position_embeddings = model.config.max_position_embeddings
        num_labels = model.config.num_labels
        orig_vocab_size = model.config.vocab_size
        if 'maskedlm' in model.__class__.__name__.lower():
            pretraining_flag = True
        if self.seq_length is not None:
            if self.encoder_seq_length is not None:
                warnings.warn('Both `seq_length` and `encoder_seq_length` are set. Using `encoder_seq_length`.')
            self.seq_length = self.encoder_seq_length
        elif self.encoder_seq_length is not None:
            self.seq_length = self.encoder_seq_length
        elif batch_data is not None:
            self.seq_length = batch_data['input_ids'].shape[1]
        else:
            self.seq_length = max_position_embeddings
        self.megatron_lm_default_args['seq_length'] = self.seq_length
    elif 'gpt2' in model.config.model_type.lower():
        model_type_name = 'gpt'
        num_layers = model.config.n_layer
        hidden_size = model.config.n_embd
        num_attention_heads = model.config.n_head
        max_position_embeddings = model.config.n_positions
        orig_vocab_size = model.config.vocab_size
        pretraining_flag = True
        if self.seq_length is not None:
            if self.decoder_seq_length is not None:
                warnings.warn('Both `seq_length` and `decoder_seq_length` are set. Using `decoder_seq_length`.')
            self.seq_length = self.decoder_seq_length
        elif self.decoder_seq_length is not None:
            self.seq_length = self.decoder_seq_length
        elif batch_data is not None:
            self.seq_length = batch_data['input_ids'].shape[1]
        else:
            self.seq_length = max_position_embeddings
        self.megatron_lm_default_args['seq_length'] = self.seq_length
        self.megatron_lm_default_args['return_logits'] = self.return_logits
        self.megatron_lm_default_args['tokenizer_type'] = 'GPT2BPETokenizer'
    elif 't5' in model.config.model_type.lower():
        model_type_name = 't5'
        num_layers = model.config.num_layers
        hidden_size = model.config.d_model
        num_attention_heads = model.config.num_heads
        max_position_embeddings = model.config.n_positions if hasattr(model.config, 'n_positions') else 1024
        orig_vocab_size = model.config.vocab_size
        pretraining_flag = True
        if self.encoder_seq_length is None:
            if batch_data is not None:
                self.encoder_seq_length = batch_data['input_ids'].shape[1]
            else:
                self.encoder_seq_length = max_position_embeddings
        if self.decoder_seq_length is None:
            if batch_data is not None:
                self.decoder_seq_length = batch_data['labels'].shape[1]
            else:
                self.decoder_seq_length = max_position_embeddings
        self.megatron_lm_default_args['encoder_seq_length'] = self.encoder_seq_length
        self.megatron_lm_default_args['decoder_seq_length'] = self.decoder_seq_length
    else:
        raise ValueError('🤗 Accelerate Megatron-LM integration supports only BERT, GPT and T5 model. Please check the model you are using is one of those.')
    self.megatron_lm_default_args['model_type_name'] = model_type_name
    self.megatron_lm_default_args['num_layers'] = num_layers
    self.megatron_lm_default_args['hidden_size'] = hidden_size
    self.megatron_lm_default_args['num_attention_heads'] = num_attention_heads
    self.megatron_lm_default_args['max_position_embeddings'] = max_position_embeddings
    self.megatron_lm_default_args['pretraining_flag'] = pretraining_flag
    self.megatron_lm_default_args['orig_vocab_size'] = orig_vocab_size
    self.megatron_lm_default_args['model_return_dict'] = model.config.return_dict
    if model_type_name == 'bert':
        self.megatron_lm_default_args['num_labels'] = num_labels