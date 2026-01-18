import dataclasses
import math
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_pegasus_x import PegasusXConfig
def compute_global_attention_representations(self, global_q, global_k, global_v, local_k, local_v, mask, dim: DimensionInfo):
    """Compute attention representations for global tokens.

        Global tokens will attend to both global tokens as well as all input sequence tokens. Because the input
        sequence tokens are arranged in blocks for local attention, we unblock them and compute attention.

        Args:
            global_q (`torch.FloatTensor`) of shape [batch_size, num_heads, global_len, dim_per_head]:
                query vectors from global tokens
            global_k (`torch.FloatTensor`) of shape [batch_size, num_heads, global_len, dim_per_head]:
                key vectors from global tokens
            global_v (`torch.FloatTensor`) of shape [batch_size, num_heads, global_len, dim_per_head]:
                value vectors from global tokens
            local_k (`torch.FloatTensor`) of shape [batch_size, num_heads, padded_seq_len, dim_per_head]:
                key vectors from local tokens
            local_v (`torch.FloatTensor`) of shape [batch_size, num_heads, padded_seq_len, dim_per_head]:
                value vectors from local tokens
            mask (`torch.FloatTensor`) of shape [batch_size, padded_seq_len]: attention mask
            dim (DimensionInfo): DimensionInfo wrapper for dimensions

        Returns:
            output of shape `[batch_sizes, length, features]`. where length will be padded to a multiple of block_size
        """
    global_and_local_k = torch.cat([global_k, local_k], dim=2)
    global_and_local_v = torch.cat([global_v, local_v], dim=2)
    extended_mask = nn.functional.pad(mask, pad=(dim.global_len, 0), value=0)
    attn_weights = torch.einsum('BHGF,BHXF->BHGX', global_q, global_and_local_k)
    attn_weights = attn_weights + extended_mask[:, None, None, :]
    attn_probs = nn.functional.softmax(attn_weights, dim=-1)
    attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    attn_output = torch.einsum('BHGX,BHXF->BHGF', attn_probs, global_and_local_v)
    return (attn_output, attn_probs)