import collections.abc
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, ImageSuperResolutionOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
from .configuration_swin2sr import Swin2SRConfig
def pad_and_normalize(self, pixel_values):
    _, _, height, width = pixel_values.size()
    window_size = self.config.window_size
    modulo_pad_height = (window_size - height % window_size) % window_size
    modulo_pad_width = (window_size - width % window_size) % window_size
    pixel_values = nn.functional.pad(pixel_values, (0, modulo_pad_width, 0, modulo_pad_height), 'reflect')
    self.mean = self.mean.type_as(pixel_values)
    pixel_values = (pixel_values - self.mean) * self.img_range
    return pixel_values