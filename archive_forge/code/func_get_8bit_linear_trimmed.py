import math
import einops
import pytest
import torch
from torch import nn
import bitsandbytes as bnb
from tests.helpers import id_formatter
@staticmethod
def get_8bit_linear_trimmed(x, stochastic=False, trim_value=3.0):
    round_func = LinearFunction.round_stoachastic if stochastic else torch.round
    norm = math.sqrt(math.pi) / math.sqrt(2.0)
    std = torch.std(x)
    max1 = std * trim_value
    x = x / max1 * 127
    x = round_func(x)
    x[x > 127] = 127
    x[x < -127] = -127
    x = x / 127 * max1
    return x