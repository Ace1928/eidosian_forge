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
@pytest.mark.parametrize('seq_dim', get_test_dims(32, 512, n=3), ids=id_formatter('seq_dim'))
@pytest.mark.parametrize('hidden_dim', get_test_dims(32, 1024 * 4, n=3), ids=id_formatter('hidden_dim'))
@pytest.mark.parametrize('batch_dim', get_test_dims(2, 16, n=3), ids=id_formatter('batch_dim'))
def test_dim3_igemm(seq_dim, hidden_dim, batch_dim):
    seq_dim = seq_dim - seq_dim % 32
    hidden_dim = hidden_dim - hidden_dim % 32
    batch_dim = batch_dim - batch_dim % 2
    for i in range(25):
        A = torch.randint(-128, 127, size=(batch_dim, seq_dim, hidden_dim), device='cuda').to(torch.int8)
        B = torch.randint(-128, 127, size=(batch_dim, seq_dim, 1024), device='cuda').to(torch.int8)
        out2 = torch.einsum('bsi, bso->io', A.float(), B.float())
        iout = torch.empty(A.shape[2], B.shape[2], dtype=torch.int32, device=A.device)
        out = F.igemm(A, B, out=iout)
        torch.testing.assert_close(out.float(), out2)