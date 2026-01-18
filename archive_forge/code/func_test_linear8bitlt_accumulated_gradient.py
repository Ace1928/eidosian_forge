import math
import einops
import pytest
import torch
from torch import nn
import bitsandbytes as bnb
from tests.helpers import id_formatter
def test_linear8bitlt_accumulated_gradient():
    l1 = torch.nn.Sequential(*[bnb.nn.Linear8bitLt(32, 32).cuda().half() for i in range(2)])
    l2 = torch.nn.Sequential(*[torch.nn.Linear(32, 32).cuda().half() for i in range(2)])
    l1[0].weight.data.copy_(l2[0].weight.data)
    l1[1].weight.data.copy_(l2[1].weight.data)
    l1[0].bias.data.copy_(l2[0].bias.data)
    l1[1].bias.data.copy_(l2[1].bias.data)
    opt1 = bnb.optim.Adam32bit(l1.parameters(), lr=0.001)
    opt2 = bnb.optim.Adam32bit(l2.parameters(), lr=0.001)
    acc_steps = 10
    for i in range(10):
        b1 = torch.randn(16, 8, 32, device='cuda').half()
        o1 = l1(b1)
        o2 = l2(b1)
        loss1 = o1.mean()
        loss2 = o2.mean()
        loss1.backward()
        loss2.backward()
        if i == 2:
            assert l1[0].state.CxB is not None
            assert l1[1].state.CxB is not None
        if i > 0 and i % acc_steps == 0:
            opt1.step()
            opt1.zero_grad(True)
            opt2.step()
            opt2.zero_grad(True)
            assert_all_approx_close(l1[0].weight, l2[0].weight, rtol=1.05, atol=0.01, count=2)
            assert_all_approx_close(l1[1].weight, l2[1].weight, rtol=1.05, atol=0.01, count=2)
            l1[0].weight.data.copy_(l2[0].weight.data)
            l1[1].weight.data.copy_(l2[1].weight.data)
            l1[0].bias.data.copy_(l2[0].bias.data)
            l1[1].bias.data.copy_(l2[1].bias.data)
        else:
            torch.testing.assert_close(l1[0].weight.grad, l2[0].weight.grad, atol=0.001, rtol=0.001)
            torch.testing.assert_close(l1[1].weight.grad, l2[1].weight.grad, atol=0.001, rtol=0.001)