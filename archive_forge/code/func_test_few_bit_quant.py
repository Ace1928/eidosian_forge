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
def test_few_bit_quant():
    for bits in range(2, 9):
        for method in ['linear', 'fp8', 'dynamic', 'quantile']:
            abserrs = []
            relerrs = []
            code = None
            if method == 'linear':
                code = F.create_linear_map(True, total_bits=bits).cuda()
            elif method == 'fp8':
                ebits = math.ceil(bits / 2)
                pbits = bits - ebits - 1
                code = F.create_fp8_map(True, ebits, pbits, bits).cuda()
            elif method == 'dynamic':
                code = F.create_dynamic_map(True, bits - 0, bits).cuda()
            elif method == 'quantile':
                values = torch.randn(2048, 2048, device='cuda')
                code = F.create_quantile_map(values, bits).cuda()
            assert torch.unique(code).numel() in [2 ** bits, 2 ** bits - 1], f'bits: {bits}, method: {method}'
            assert code.numel() == 256
            for i in range(10):
                values = torch.randn(1, 32, device='cuda')
                values /= values.abs().max()
                q1 = []
                v1 = []
                for v in values[0]:
                    idx = torch.abs(v - code).argmin()
                    q1.append(idx.item())
                    v1.append(code[idx].item())
                q1 = torch.Tensor(q1).cuda()
                v1 = torch.Tensor(v1).cuda()
                q2, S2 = F.quantize_blockwise(values, code=code)
                v2 = F.dequantize_blockwise(q2, S2)
                idx = torch.isclose(q1.int(), q2.int())
                err2 = torch.abs(v2 - values)
                abserrs.append(err2.mean().item())
                relerrs.append((err2 / (1e-10 + values).abs()).mean().item())
                if idx.sum():
                    err1 = torch.abs(v1 - values).mean()
                else:
                    torch.testing.assert_close(q1, q2)