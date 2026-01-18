import copy
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_clipseg import CLIPSegConfig, CLIPSegTextConfig, CLIPSegVisionConfig
def interpolate_position_embeddings(self, new_size):
    if len(new_size) != 2:
        raise ValueError('new_size should consist of 2 values')
    num_patches_one_direction = int(self.num_patches ** 0.5)
    a = self.position_embedding.weight[1:].T.view(1, self.config.hidden_size, num_patches_one_direction, num_patches_one_direction)
    b = nn.functional.interpolate(a, new_size, mode='bicubic', align_corners=False).squeeze(0).view(self.config.hidden_size, new_size[0] * new_size[1]).T
    result = torch.cat([self.position_embedding.weight[:1], b])
    return result