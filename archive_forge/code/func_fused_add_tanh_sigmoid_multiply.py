import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_vits import VitsConfig
@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, num_channels):
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :num_channels, :])
    s_act = torch.sigmoid(in_act[:, num_channels:, :])
    acts = t_act * s_act
    return acts