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
def fft_convolution(self, inputs, kernel, length):
    inputs_fft = torch.fft.rfft(inputs.float(), n=2 * length)
    kernel_fft = torch.fft.rfft(kernel.float(), n=2 * length)
    convolved_sequence = torch.fft.irfft(inputs_fft * kernel_fft, n=2 * length)
    return convolved_sequence