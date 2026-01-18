from itertools import product
import math
import random
import time
import einops
import numpy as np
import pytest
from scipy.stats import norm
import torch
import bitsandbytes as bnb
from bitsandbytes import functional as F
from tests.helpers import (
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16], ids=describe_dtype)
def test_fp4_quant(dtype):
    vals = list(product([0, 1], repeat=4))
    code = {}
    for bits in vals:
        result = 0
        bias = 3
        sign, e1, e2, p1 = bits
        idx = sign * 8 + e1 * 4 + e2 * 2 + p1 * 1
        sign = -1.0 if sign else 1.0
        exp = e1 * 2 + e2 * 1
        if exp == 0:
            if p1 == 0:
                result = 0
            else:
                result = sign * 0.0625
        else:
            exp = 2 ** (-exp + bias + 1)
            frac = 1.5 if p1 else 1.0
            result = sign * exp * frac
        code[idx] = result
    A1 = torch.randn(1024, 1024, device='cuda', dtype=dtype)
    qa, SA = F.quantize_fp4(A1, blocksize=64)
    A2 = F.dequantize_fp4(qa, SA)
    err = (A1 - A2).abs().float()
    relerr = (err / (A1.abs().float() + 1e-08)).mean()
    idx = err > 1.0
    err = err.mean()
    assert A2.dtype == dtype
    assert err.item() < 0.1
    assert relerr.item() < 0.28