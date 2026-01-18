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
@pytest.mark.parametrize('seq_dim', get_test_dims(32, 512, n=2), ids=id_formatter('seq_dim'))
@pytest.mark.parametrize('hidden_dim', get_test_dims(32, 1024 * 4, n=2), ids=id_formatter('hidden_dim'))
@pytest.mark.parametrize('batch_dim', get_test_dims(2, 16, n=2), ids=id_formatter('batch_dim'))
@pytest.mark.parametrize('transpose', TRUE_FALSE, ids=id_formatter('transpose'))
def test_minmax_igemm(seq_dim, hidden_dim, batch_dim, transpose):

    def min_max(x):
        maxA = torch.amax(x, dim=2, keepdim=True)
        minA = torch.amin(x, dim=2, keepdim=True)
        scale = (maxA - minA) / 2.0
        return ((127 * (x - minA - scale) / scale).to(torch.int8), minA, scale)
    seq_dim = seq_dim - seq_dim % 16
    hidden_dim = hidden_dim - hidden_dim % 16
    batch_dim = batch_dim - batch_dim % 2
    errs = []
    relerrs = []
    errs2 = []
    relerrs2 = []
    for i in range(k):
        A = torch.normal(0.0, 0.5, size=(batch_dim, seq_dim, hidden_dim), device='cuda')
        if transpose:
            B = torch.normal(0, 0.5, size=(256, hidden_dim), device='cuda')
        else:
            B = torch.normal(0, 0.5, size=(hidden_dim, 256), device='cuda')
        Ac, minA, scale = min_max(A)
        if transpose:
            maxB, Bc = quant_multi(B, dim=1 if transpose else 0)
            out = F.igemm(Ac, Bc.t())
            out2 = torch.matmul(A, B.t())
            offset = B.t().sum(0) * (minA + scale)
            out = out.float()
            out = out * maxB.t() * scale / (127 * 127) + offset
            maxA, Ac = quant_multi(A, dim=2)
            out3 = F.igemm(Ac, Bc.t())
            out3 = mm_dequant(maxA, maxB.t(), out3)
        else:
            maxB, Bc = quant_multi(B, dim=0)
            offset = B.sum(0) * (minA + scale)
            out = F.igemm(Ac, Bc)
            out2 = torch.matmul(A, B)
            out = out.float()
            out = out * maxB * scale / (127 * 127) + offset
            maxA, Ac = quant_multi(A, dim=2)
            out3 = F.igemm(Ac, Bc)
            out3 = mm_dequant(maxA, maxB, out3)
        std = out2.std()
        out2 /= std
        out /= std
        out3 /= std
        err = torch.abs(out - out2)
        relerr = err / (torch.abs(out2) + 1e-07)
        err2 = torch.abs(out3 - out2)
        relerr2 = err2 / (torch.abs(out2) + 1e-07)
        errs.append(err.mean().item())
        relerrs.append(relerr.mean().item())
        errs2.append(err2.mean().item())
        relerrs2.append(relerr2.mean().item())
    assert mean(errs) < 0.015
    assert mean(relerrs) < 0.3