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
def test_coo2csr():
    threshold = 1
    A = torch.randn(128, 128).half().cuda()
    idx = torch.abs(A) >= threshold
    nnz = (idx == 1).sum().item()
    rows, cols = torch.where(idx)
    values = A[idx]
    cooA = F.COOSparseTensor(A.shape[0], A.shape[1], nnz, rows.int(), cols.int(), values)
    A2 = A * idx
    csrA = F.coo2csr(cooA)
    counts = csrA.rowptr[1:] - csrA.rowptr[:-1]
    assert counts.numel() == A.shape[0]
    torch.testing.assert_close(counts.long(), (A2 != 0).sum(1))
    idx = A2 != 0
    torch.testing.assert_close(A2[idx], csrA.values)