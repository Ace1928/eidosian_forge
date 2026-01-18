import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
from .configuration_mega import MegaConfig
def get_ema_kernel(self, length: int):
    kernel_size = length if self.truncation is None else min(self.truncation, length)
    if self.training:
        return self._compute_efficient_ema_kernel(kernel_size)
    else:
        if self._kernel is None or self._kernel.size(-1) < kernel_size:
            self._kernel = self._compute_efficient_ema_kernel(kernel_size)
        return self._kernel[..., :kernel_size]