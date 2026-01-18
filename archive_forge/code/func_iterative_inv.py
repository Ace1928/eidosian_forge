import math
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_nystromformer import NystromformerConfig
def iterative_inv(self, mat, n_iter=6):
    identity = torch.eye(mat.size(-1), device=mat.device)
    key = mat
    if self.init_option == 'original':
        value = 1 / torch.max(torch.sum(key, dim=-2)) * key.transpose(-1, -2)
    else:
        value = 1 / torch.max(torch.sum(key, dim=-2), dim=-1).values[:, :, None, None] * key.transpose(-1, -2)
    for _ in range(n_iter):
        key_value = torch.matmul(key, value)
        value = torch.matmul(0.25 * value, 13 * identity - torch.matmul(key_value, 15 * identity - torch.matmul(key_value, 7 * identity - key_value)))
    return value