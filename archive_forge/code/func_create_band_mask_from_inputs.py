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