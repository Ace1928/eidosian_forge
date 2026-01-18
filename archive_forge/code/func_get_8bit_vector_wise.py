import math
import einops
import pytest
import torch
from torch import nn
import bitsandbytes as bnb
from tests.helpers import id_formatter
@staticmethod
def get_8bit_vector_wise(x, dim, stochastic=False):
    round_func = LinearFunction.round_stoachastic if stochastic else torch.round
    max1 = torch.amax(torch.abs(x), dim=dim, keepdim=True)
    max1[max1 == 0] = 1.0
    x = x * 127 / max1
    x = round_func(x) / 127 * max1
    return x