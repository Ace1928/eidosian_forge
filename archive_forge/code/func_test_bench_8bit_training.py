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
@pytest.mark.parametrize(('batch', 'seq', 'model', 'hidden'), [pytest.param(2, 512, 4 * 1024, 3 * 4 * 1024, id='batch=2, seq=512, model=4k, hidden=12k'), pytest.param(2, 512, 5120, 3 * 5120, id='batch=2, seq=512, model=5k, hidden=15k'), pytest.param(2, 512, 12 * 1024, 4 * 12 * 1024, id='batch=2, seq=512, model=12k, hidden=48k')])
@pytest.mark.benchmark
def test_bench_8bit_training(batch, seq, model, hidden):
    formatB = F.get_special_format_str()
    A = torch.randn(batch, seq, model, device='cuda').half()
    grad = torch.randn(batch, seq, model, device='cuda').half()
    w1 = torch.randint(-128, 127, size=(hidden, model), device='cuda').half()
    w2 = torch.randint(-128, 127, size=(model, hidden), device='cuda').half()
    print('')
    dtype = torch.int8
    A = A.view(-1, A.shape[-1]).contiguous()
    grad = grad.view(-1, grad.shape[-1]).contiguous()
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(k):
        out1 = torch.matmul(A, w1.t())
    torch.cuda.synchronize()
    t16 = time.time() - t0
    print(t16)