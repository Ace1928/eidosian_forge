import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
from .configuration_big_bird import BigBirdConfig
@staticmethod
def create_masks_for_block_sparse_attn(attention_mask: torch.Tensor, block_size: int):
    batch_size, seq_length = attention_mask.size()
    if seq_length % block_size != 0:
        raise ValueError(f'Sequence length must be multiple of block size, but sequence length is {seq_length}, while block size is {block_size}.')

    def create_band_mask_from_inputs(from_blocked_mask, to_blocked_mask):
        """
            Create 3D attention mask from a 2D tensor mask.

            Args:
                from_blocked_mask: 2D Tensor of shape [batch_size,
                from_seq_length//from_block_size, from_block_size].
                to_blocked_mask: int32 Tensor of shape [batch_size,
                to_seq_length//to_block_size, to_block_size].

            Returns:
                float Tensor of shape [batch_size, 1, from_seq_length//from_block_size-4, from_block_size,
                3*to_block_size].
            """
        exp_blocked_to_pad = torch.cat([to_blocked_mask[:, 1:-3], to_blocked_mask[:, 2:-2], to_blocked_mask[:, 3:-1]], dim=2)
        band_mask = torch.einsum('blq,blk->blqk', from_blocked_mask[:, 2:-2], exp_blocked_to_pad)
        band_mask.unsqueeze_(1)
        return band_mask
    blocked_encoder_mask = attention_mask.view(batch_size, seq_length // block_size, block_size)
    band_mask = create_band_mask_from_inputs(blocked_encoder_mask, blocked_encoder_mask)
    from_mask = attention_mask.view(batch_size, 1, seq_length, 1)
    to_mask = attention_mask.view(batch_size, 1, 1, seq_length)
    return (blocked_encoder_mask, band_mask, from_mask, to_mask)