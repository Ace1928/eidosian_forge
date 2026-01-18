import collections
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
from .configuration_clap import ClapAudioConfig, ClapConfig, ClapTextConfig
def set_shift_and_window_size(self, input_resolution):
    if min(input_resolution) <= self.window_size:
        self.shift_size = 0
        self.window_size = min(input_resolution)