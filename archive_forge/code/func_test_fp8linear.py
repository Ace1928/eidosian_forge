import math
import einops
import pytest
import torch
from torch import nn
import bitsandbytes as bnb
from tests.helpers import id_formatter
def test_fp8linear():
    b = 10
    h = 1024
    inp = torch.randn(b, h).cuda()
    fp32 = torch.nn.Linear(h, h * 2).cuda()
    fp8 = bnb.research.nn.LinearFP8Mixed(h, h * 2).cuda()
    fp32b = torch.nn.Linear(h * 2, h).cuda()
    fp8b = bnb.research.nn.LinearFP8Mixed(h * 2, h).cuda()
    fp8.weight.data.copy_(fp32.weight.data)
    fp8.bias.data.copy_(fp32.bias.data)
    fp8b.weight.data.copy_(fp32b.weight.data)
    fp8b.bias.data.copy_(fp32b.bias.data)
    a = fp32b(torch.nn.functional.gelu(fp32(inp)))
    b = fp8b(torch.nn.functional.gelu(fp8(inp)))
    err = (a - b).abs().mean()
    a.mean().backward()
    b.mean().backward()
    graderr = (fp8.weight.grad - fp32.weight.grad).abs().mean()
    bgraderr = (fp8.bias.grad - fp32.bias.grad).abs().mean()
    assert err < 0.05
    assert graderr < 2e-05
    assert bgraderr < 2e-05