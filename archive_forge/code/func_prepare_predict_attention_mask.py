import copy
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import LayerNorm
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_xlm_prophetnet import XLMProphetNetConfig
def prepare_predict_attention_mask(self, hidden_states, attention_mask):
    batch_size, seq_length = hidden_states.shape[:2]
    predict_causal_mask = ngram_attention_bias(self.max_target_positions, self.ngram, hidden_states.device, hidden_states.dtype)
    predict_causal_mask = torch.cat([predict_causal_mask[:, :seq_length, :seq_length], predict_causal_mask[:, :seq_length, self.max_target_positions:self.max_target_positions + seq_length]], dim=-1)
    extended_predict_causal_mask = predict_causal_mask[None, None, :, :, :].expand((batch_size, self.config.num_decoder_attention_heads) + predict_causal_mask.shape)
    if attention_mask is not None:
        extended_attention_mask = (1.0 - attention_mask[:, None, None, None, :]) * torch.finfo(self.dtype).min
        extended_attention_mask = extended_attention_mask.expand((batch_size, self.config.num_decoder_attention_heads, self.ngram, seq_length, seq_length))
        extended_attention_mask = torch.cat([extended_attention_mask, torch.zeros_like(extended_attention_mask)], dim=-1)
        extended_predict_attention_mask = extended_predict_causal_mask + extended_attention_mask
    else:
        extended_predict_attention_mask = extended_predict_causal_mask
    return extended_predict_attention_mask.to(hidden_states.dtype)