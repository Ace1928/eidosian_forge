import math
import einops
import pytest
import torch
from torch import nn
import bitsandbytes as bnb
from tests.helpers import id_formatter
@staticmethod
def round_stoachastic(x):
    sign = torch.sign(x)
    absx = torch.abs(x)
    decimal = absx - torch.floor(absx)
    rdm = torch.rand_like(decimal)
    return sign * (torch.floor(absx) + (rdm < decimal).to(x.dtype))