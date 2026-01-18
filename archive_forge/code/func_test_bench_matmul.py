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
@pytest.mark.parametrize(('batch', 'seq', 'model', 'hidden'), [pytest.param(1, 1, 6656, 4 * 6656, id='batch=1, seq=1, model=6656, hidden=26k')])
@pytest.mark.benchmark
def test_bench_matmul(batch, seq, model, hidden):
    iters = 1000
    formatB = F.get_special_format_str()
    A = torch.randn(batch, seq, model, device='cuda').half()
    B = torch.empty(hidden, model, dtype=torch.float16, device='cuda')
    torch.nn.init.xavier_uniform_(B)
    B_fp4, state = F.quantize_fp4(B)
    B_fp4_c, state_c = F.quantize_fp4(B, compress_statistics=True)
    B_nf4, state_nf4 = F.quantize_nf4(B)
    B_nf4_c, state_nf4_c = F.quantize_nf4(B, compress_statistics=True)
    linear8bit = bnb.nn.Linear8bitLt(model, hidden, False, False).cuda().half()
    linear8bit.eval()
    outliers = torch.randint(0, model, size=(5,)).cuda()
    A[:, :, outliers] = 8.0
    linearMixedBit = bnb.nn.Linear8bitLt(model, hidden, False, False, threshold=6.0).cuda().half()
    linear8bit_train = bnb.nn.Linear8bitLt(model, hidden, False).cuda().half()
    linear8bit_train_thresh = bnb.nn.Linear8bitLt(model, hidden, False, threshold=6.0).cuda().half()
    bnb.matmul_4bit(A, B_nf4.t(), quant_state=state_nf4)
    for i in range(iters):
        torch.matmul(A, B.t())
    torch.cuda.synchronize()
    print('')
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(iters):
        torch.matmul(A, B.t())
    torch.cuda.synchronize()
    print(f'pytorch fp16: [{batch},{seq},{model}], [{model},{hidden}]->[{batch},{seq},{hidden}]: {time.time() - t0:.4f}s')
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(iters):
        bnb.matmul_4bit(A, B_nf4.t(), quant_state=state_nf4)
    torch.cuda.synchronize()
    print(f'bnb nf4: [{batch},{seq},{model}], [{model},{hidden}]->[{batch},{seq},{hidden}]: {time.time() - t0:.4f}s')
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(iters):
        bnb.matmul_4bit(A, B_nf4_c.t(), quant_state=state_nf4_c)
    torch.cuda.synchronize()
    print(f'bnb nf4+DQ: [{batch},{seq},{model}], [{model},{hidden}]->[{batch},{seq},{hidden}]: {time.time() - t0:.4f}s')