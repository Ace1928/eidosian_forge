from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_mobilevitv2 import MobileViTV2Config
def folding(self, patches: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
    batch_size, in_dim, patch_size, n_patches = patches.shape
    patches = patches.reshape(batch_size, in_dim * patch_size, n_patches)
    feature_map = nn.functional.fold(patches, output_size=output_size, kernel_size=(self.patch_height, self.patch_width), stride=(self.patch_height, self.patch_width))
    return feature_map