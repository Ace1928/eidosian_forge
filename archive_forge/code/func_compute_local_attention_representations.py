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
def compute_local_attention_representations(self, global_k, global_v, local_q, local_k, local_v, mask, dim: DimensionInfo):
    """Compute attention representations for local tokens.

        Local tokens will attend to both global tokens as well as all other tokens within the same local block. Hence,
        we need to tile and concatenate the global tokens to every local block

        Args:
            global_k (`torch.FloatTensor`) of shape [batch_size, num_heads, global_len, dim_per_head]:
                key vectors from global tokens
            global_v (`torch.FloatTensor`) of shape [batch_size, num_heads, global_len, dim_per_head]:
                value vectors from global tokens
            local_q (`torch.FloatTensor`) of shape [batch_size, num_heads, padded_seq_len, dim_per_head]:
                query vectors from local tokens
            local_k (`torch.FloatTensor`) of shape [batch_size, num_heads, padded_seq_len, dim_per_head]:
                key vectors from local tokens
            local_v (`torch.FloatTensor`) of shape [batch_size, num_heads, padded_seq_len, dim_per_head]:
                value vectors from local tokens
            mask (`torch.FloatTensor`) of shape [batch_size, padded_seq_len]: attention mask
            dim (DimensionInfo): DimensionInfo wrapper for dimensions

        Returns:
            output of shape `[batch_sizes, length, features]`. where length will be padded to a multiple of block_size
        """
    blocked_local_q = local_q.view(dim.batch_size, dim.num_heads, dim.num_blocks, dim.block_size, dim.dim_per_head)
    blocked_local_k = local_k.view(dim.batch_size, dim.num_heads, dim.num_blocks, dim.block_size, dim.dim_per_head)
    blocked_local_v = local_v.view(dim.batch_size, dim.num_heads, dim.num_blocks, dim.block_size, dim.dim_per_head)
    extended_mask = nn.functional.pad(mask.view(dim.batch_size, dim.num_blocks, dim.block_size), pad=(dim.global_len, 0), value=0)
    blocked_local2global = torch.einsum('BHNKF,BHGF->BHNKG', blocked_local_q, global_k)
    blocked_local2local = torch.einsum('BHNKF,BHNXF->BHNKX', blocked_local_q, blocked_local_k)
    attn_weights = torch.cat([blocked_local2global, blocked_local2local], dim=-1)
    attn_weights = attn_weights + extended_mask[:, None, :, None, :]
    attn_probs = nn.functional.softmax(attn_weights, dim=-1)
    attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    local2global_attn_probs = attn_probs[:, :, :, :, :dim.global_len]
    local2local_attn_probs = attn_probs[:, :, :, :, dim.global_len:]
    local2global_attn_output = torch.einsum('BHNKG,BHGF->BHNKF', local2global_attn_probs, global_v)
    local2local_attn_output = torch.einsum('BHNKX,BHNXF->BHNKF', local2local_attn_probs, blocked_local_v)
    attn_output = local2global_attn_output + local2local_attn_output
    return (attn_output, attn_probs)