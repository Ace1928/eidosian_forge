from collections.abc import Sequence
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import softmax_backward_data
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_deberta_v2 import DebertaV2Config
@staticmethod
def backward(ctx, grad_output):
    if ctx.scale > 1:
        mask, = ctx.saved_tensors
        return (grad_output.masked_fill(mask, 0) * ctx.scale, None)
    else:
        return (grad_output, None)