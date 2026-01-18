import math
import einops
import pytest
import torch
from torch import nn
import bitsandbytes as bnb
from tests.helpers import id_formatter
def get_8bit_linear(x, stochastic=False):
    round_func = LinearFunction.round_stoachastic if stochastic else torch.round
    max1 = torch.abs(x).max()
    x = x / max1 * 127
    x = round_func(x) / 127 * max1
    return x