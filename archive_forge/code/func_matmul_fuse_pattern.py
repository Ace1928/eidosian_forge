import functools
import torch
from torch._inductor.compile_fx import fake_tensor_prop
from ..._dynamo.utils import counters
from .. import config
from ..pattern_matcher import (
def matmul_fuse_pattern(inp, w1, w2, w3):
    return (inp @ w1, inp @ w2, inp @ w3)