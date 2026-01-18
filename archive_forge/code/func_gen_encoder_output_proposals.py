import copy
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from ...activations import ACT2FN
from ...file_utils import (
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import meshgrid
from ...utils import is_accelerate_available, is_ninja_available, is_torchvision_available, logging, requires_backends
from ...utils.backbone_utils import load_backbone
from .configuration_deta import DetaConfig
def gen_encoder_output_proposals(self, enc_output, padding_mask, spatial_shapes):
    """Generate the encoder output proposals from encoded enc_output.

        Args:
            enc_output (Tensor[batch_size, sequence_length, hidden_size]): Output of the encoder.
            padding_mask (Tensor[batch_size, sequence_length]): Padding mask for `enc_output`.
            spatial_shapes (Tensor[num_feature_levels, 2]): Spatial shapes of the feature maps.

        Returns:
            `tuple(torch.FloatTensor)`: A tuple of feature map and bbox prediction.
                - object_query (Tensor[batch_size, sequence_length, hidden_size]): Object query features. Later used to
                  directly predict a bounding box. (without the need of a decoder)
                - output_proposals (Tensor[batch_size, sequence_length, 4]): Normalized proposals, after an inverse
                  sigmoid.
        """
    batch_size = enc_output.shape[0]
    proposals = []
    _cur = 0
    level_ids = []
    for level, (height, width) in enumerate(spatial_shapes):
        mask_flatten_ = padding_mask[:, _cur:_cur + height * width].view(batch_size, height, width, 1)
        valid_height = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
        valid_width = torch.sum(~mask_flatten_[:, 0, :, 0], 1)
        grid_y, grid_x = meshgrid(torch.linspace(0, height - 1, height, dtype=torch.float32, device=enc_output.device), torch.linspace(0, width - 1, width, dtype=torch.float32, device=enc_output.device), indexing='ij')
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
        scale = torch.cat([valid_width.unsqueeze(-1), valid_height.unsqueeze(-1)], 1).view(batch_size, 1, 1, 2)
        grid = (grid.unsqueeze(0).expand(batch_size, -1, -1, -1) + 0.5) / scale
        width_heigth = torch.ones_like(grid) * 0.05 * 2.0 ** level
        proposal = torch.cat((grid, width_heigth), -1).view(batch_size, -1, 4)
        proposals.append(proposal)
        _cur += height * width
        level_ids.append(grid.new_ones(height * width, dtype=torch.long) * level)
    output_proposals = torch.cat(proposals, 1)
    output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
    output_proposals = torch.log(output_proposals / (1 - output_proposals))
    output_proposals = output_proposals.masked_fill(padding_mask.unsqueeze(-1), float('inf'))
    output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))
    object_query = enc_output
    object_query = object_query.masked_fill(padding_mask.unsqueeze(-1), float(0))
    object_query = object_query.masked_fill(~output_proposals_valid, float(0))
    object_query = self.enc_output_norm(self.enc_output(object_query))
    level_ids = torch.cat(level_ids)
    return (object_query, output_proposals, level_ids)