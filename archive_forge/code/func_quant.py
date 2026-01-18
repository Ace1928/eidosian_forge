import math
import einops
import pytest
import torch
from torch import nn
import bitsandbytes as bnb
from tests.helpers import id_formatter
def quant(x, quant_type, dim=1):
    if quant_type == 'linear':
        max1 = torch.abs(x).max().float()
        xq = torch.round(x / max1 * 127).to(torch.int8)
        return (xq, max1)
    elif quant_type == 'vector':
        max1 = torch.amax(torch.abs(x), dim=dim, keepdim=True)
        xq = torch.round(x / max1 * 127).to(torch.int8)
        return (xq, max1)
    elif quant_type == 'min-max':
        maxA = torch.amax(x, dim=dim, keepdim=True).float()
        minA = torch.amin(x, dim=dim, keepdim=True).float()
        scale = (maxA - minA) / 2.0
        xq = torch.round(127 * (x - minA - scale) / scale).to(torch.int8)
        return (xq, (minA.float(), scale.float()))
    else:
        return None