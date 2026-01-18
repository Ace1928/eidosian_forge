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
def test_matmuls():
    a = torch.randn(256, 512).half().cuda()
    b = torch.randn(256, 512).half().cuda()
    c1 = torch.matmul(a, b.t())
    c2 = bnb.matmul(a, b)
    c3 = bnb.matmul_cublas(a, b.t())
    err1 = torch.abs(c1 - c2).mean().item()
    err2 = torch.abs(c1 - c3).mean().item()
    assert err1 < 0.2
    assert err2 < 0.2
    print(err1, err2)