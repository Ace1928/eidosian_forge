import os
from os.path import join
import shutil
import time
import uuid
from lion_pytorch import Lion
import pytest
import torch
import bitsandbytes as bnb
import bitsandbytes.functional as F
from tests.helpers import describe_dtype, id_formatter
@pytest.mark.parametrize('dim1', [4096], ids=id_formatter('dim1'))
@pytest.mark.parametrize('dim2', [4096], ids=id_formatter('dim2'))
@pytest.mark.parametrize('gtype', [torch.float32, torch.float16], ids=describe_dtype)
@pytest.mark.parametrize('optim_name', optimizer_names_benchmark, ids=id_formatter('opt'))
@pytest.mark.benchmark
def test_benchmark_blockwise(dim1, dim2, gtype, optim_name):
    if dim1 == 1 and dim2 == 1:
        return
    p1 = torch.randn(dim1, dim2, device='cuda', dtype=gtype) * 0.1
    bnb_optimizer = str2optimizers[optim_name][1]([p1])
    g = torch.randn(dim1, dim2, device='cuda', dtype=gtype) * 0.01
    p1.grad = g
    for i in range(k):
        if i == k // 5:
            torch.cuda.synchronize()
            t0 = time.time()
        bnb_optimizer.step()
    torch.cuda.synchronize()
    s = time.time() - t0
    print('')
    params = (k - k // 5) * dim1 * dim2
    print(optim_name, gtype, s / params)