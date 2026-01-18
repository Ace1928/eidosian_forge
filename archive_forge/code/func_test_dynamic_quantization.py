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
def test_dynamic_quantization():
    diffs = []
    reldiffs = []
    for i in range(100):
        A1 = torch.randn(1024, 1024, device='cuda')
        C, S = F.quantize(A1)
        A2 = F.dequantize(C, S)
        diff = torch.abs(A1 - A2)
        reldiff = diff / torch.abs(A1 + 1e-08)
        diffs.append(diff.mean().item())
        reldiffs.append(reldiff.mean().item())
        assert diff.mean().item() < 0.0135
    print(sum(diffs) / len(diffs))
    print(sum(reldiffs) / len(reldiffs))
    for i in range(100):
        A1 = torch.rand(1024, 1024, device='cuda')
        C, S = F.quantize(A1)
        A2 = F.dequantize(C, S)
        diff = torch.abs(A1 - A2).mean().item()
        torch.testing.assert_close(A1, A2, atol=0.01, rtol=0)
        assert diff < 0.004