import collections.abc
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_tvlt import TvltConfig
def patchify_audio(self, audio_values):
    """
        audio_values: [batch_size, 1, height, width]
        """
    batch_size, num_channels, height, width = audio_values.shape
    num_patches_height = height // self.audio_patch_size[0]
    num_patches_width = width // self.audio_patch_size[1]
    patchified_audio_values = audio_values.reshape(shape=(batch_size, num_channels, num_patches_height, self.audio_patch_size[0], num_patches_width, self.audio_patch_size[1]))
    patchified_audio_values = torch.einsum('nchpwq->nhwpqc', patchified_audio_values)
    patchified_audio_values = patchified_audio_values.reshape(shape=(batch_size, num_patches_height * num_patches_width, self.audio_patch_size[0] * self.audio_patch_size[1] * num_channels))
    return patchified_audio_values