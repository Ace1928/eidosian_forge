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
@pytest.mark.parametrize('nested', TRUE_FALSE, ids=id_formatter('nested'))
@pytest.mark.parametrize('blocksize', [4096, 2048, 1024, 512, 256, 128, 64])
@pytest.mark.parametrize('signed', TRUE_FALSE, ids=id_formatter('signed'))
def test_dynamic_blockwise_quantization(dtype, nested, blocksize, signed):
    diffs = []
    reldiffs = []
    for i in range(100):
        A1 = torch.randn(1024, 1024, device='cuda', dtype=dtype)
        C, S = F.quantize_blockwise(A1, blocksize=blocksize, nested=nested)
        A2 = F.dequantize_blockwise(C, S)
        diff = torch.abs(A1 - A2).float()
        reldiff = diff / torch.abs(A1.float() + 1e-08)
        diffs.append(diff.mean().item())
        reldiffs.append(reldiff.mean().item())
    abserr = sum(diffs) / len(diffs)
    relerr = sum(reldiffs) / len(reldiffs)
    assert abserr < 0.011
    assert relerr < 0.018
    assert A2.dtype == dtype
    diffs = []
    code = F.create_dynamic_map(signed=signed)
    for i in range(100):
        A1 = torch.rand(1024, 1024, device='cuda', dtype=dtype)
        C, S = F.quantize_blockwise(A1, blocksize=blocksize, nested=nested, code=code)
        A2 = F.dequantize_blockwise(C, S)
        diff = torch.abs(A1 - A2).float()
        reldiff = diff / torch.abs(A1.float() + 1e-08)
        diffs.append(diff.mean().item())
        reldiffs.append(reldiff.mean().item())
    abserr = sum(diffs) / len(diffs)
    relerr = sum(reldiffs) / len(reldiffs)
    if signed:
        assert abserr < 0.0035
        assert relerr < 0.015
    else:
        assert abserr < 0.00175
        assert relerr < 0.012
    assert A2.dtype == dtype