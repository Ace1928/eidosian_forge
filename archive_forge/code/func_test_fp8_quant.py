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
def test_fp8_quant():
    for e_bits in range(1, 7):
        p_bits = 7 - e_bits
        code = F.create_fp8_map(True, e_bits, p_bits).cuda()
        abserr = []
        relerr = []
        for i in range(100):
            A1 = torch.randn(1024, 1024, device='cuda')
            C, SC = F.quantize_blockwise(A1, code=code)
            A2 = F.dequantize_blockwise(C, SC)
            diff = torch.abs(A1 - A2)
            reldiff = diff / torch.abs(A1 + 1e-08)
            abserr.append(diff.mean().item())
            relerr.append(reldiff.mean().item())
        abserr = []
        relerr = []
        for i in range(100):
            A1 = torch.rand(1024, 1024, device='cuda')
            C, SC = F.quantize_blockwise(A1, code=code)
            A2 = F.dequantize_blockwise(C, SC)
            diff = torch.abs(A1 - A2)
            reldiff = diff / torch.abs(A1 + 1e-08)
            abserr.append(diff.mean().item())
            relerr.append(reldiff.mean().item())
        abserr = []
        relerr = []
        for i in range(100):
            A1 = torch.randn(1024, 1024, device='cuda')
            C, SC = F.quantize_blockwise(A1)
            A2 = F.dequantize_blockwise(C, SC)
            diff = torch.abs(A1 - A2)
            reldiff = diff / torch.abs(A1 + 1e-08)
            abserr.append(diff.mean().item())
            relerr.append(reldiff.mean().item())