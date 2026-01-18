import math
import einops
import pytest
import torch
from torch import nn
import bitsandbytes as bnb
from tests.helpers import id_formatter
@staticmethod
def fake_8bit_storage(w, exponent_bits):
    code = bnb.functional.create_dynamic_map(n=exponent_bits).to(w.device)
    absmax, C = bnb.functional.quantize_blockwise(w.data, code=code)
    out = bnb.functional.dequantize_blockwise(absmax, C, code)
    out = out.half()
    w.copy_(out)
    return out