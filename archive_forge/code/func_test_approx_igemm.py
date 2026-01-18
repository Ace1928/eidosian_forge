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
@pytest.mark.parametrize('dim1', [1024 * 2], ids=id_formatter('dim1'))
@pytest.mark.parametrize('dim2', [1024 * 16], ids=id_formatter('dim2'))
@pytest.mark.parametrize('quant_methods', methods.values(), ids=methods.keys())
@pytest.mark.parametrize('batched', TRUE_FALSE, ids=id_formatter('batched'))
def test_approx_igemm(dim1, dim2, quant_methods, batched):
    dim1 = dim1 - dim1 % 32
    dim2 = dim2 - dim2 % 32
    errors = []
    relerrors = []
    for i in range(5):
        if batched:
            A = torch.normal(0, 0.5, size=(32, dim1, dim2 // 32), device='cuda')
            B = torch.normal(0, 0.5, size=(32, dim2 // 32, dim1), device='cuda')
            maxA, Ac = quant_methods[0](A, 2)
            maxB, Bc = quant_methods[1](B, 1)
        else:
            A = torch.normal(0, 0.5, size=(dim1, dim2), device='cuda')
            B = torch.normal(0, 0.5, size=(dim2, dim1), device='cuda')
            maxA, Ac = quant_methods[0](A, 1)
            maxB, Bc = quant_methods[1](B, 0)
        torch.testing.assert_close(quant_methods[2](maxA, Ac), A, atol=0.025, rtol=0.05)
        if batched:
            out2 = torch.bmm(A, B)
            C = torch.bmm(Ac.float(), Bc.float())
        else:
            out2 = torch.mm(A, B)
            C = F.igemm(Ac, Bc)
        out = quant_methods[4](maxA, maxB, C)
        std = out2.std()
        out /= std
        out2 /= std
        err = torch.abs(out - out2)
        relerr = err / torch.abs(out2)
        errors.append(err.mean().item())
        relerrors.append(relerr.mean().item())