import math
import einops
import pytest
import torch
from torch import nn
import bitsandbytes as bnb
from tests.helpers import id_formatter
@pytest.mark.parametrize('threshold', [0.0, 2.0])
@pytest.mark.parametrize('memory_efficient_backward', [False])
def test_linear8bitlt_no_fp16_weights(threshold, memory_efficient_backward):
    l1 = bnb.nn.Linear8bitLt(32, 64, threshold=threshold, has_fp16_weights=False, memory_efficient_backward=memory_efficient_backward).cuda().half()
    assert l1.weight.dtype == torch.int8
    l1.eval()
    for i in range(100):
        b1 = torch.randn(16, 8, 32, device='cuda').half()
        o1 = l1(b1)
        assert o1.dtype == torch.float16
    mlp = MLP8bit(32, 64, threshold=threshold, has_fp16_weights=False).cuda()
    assert mlp.fc1.weight.dtype == torch.int8
    assert mlp.fc2.weight.dtype == torch.int8
    for i in range(100):
        b1 = torch.randn(16, 8, 32, device='cuda').half()
        o1 = mlp(b1)
        assert o1.dtype == torch.float16
        if threshold > 0:
            assert mlp.fc1.state.idx is not None
        if threshold > 0:
            assert mlp.fc2.state.idx is not None
    mlp = MLP8bit(32, 64, threshold=threshold, has_fp16_weights=False).cuda().half()
    assert mlp.fc1.weight.dtype == torch.int8
    assert mlp.fc2.weight.dtype == torch.int8
    for i in range(100):
        b1 = torch.randn(16, 8, 32, device='cuda').half()
        o1 = mlp(b1)
        assert o1.dtype == torch.float16
        if threshold > 0:
            assert mlp.fc1.state.idx is not None
        if threshold > 0:
            assert mlp.fc2.state.idx is not None
    mlp = MLP8bit(32, 64, threshold=threshold, has_fp16_weights=False).half().cuda()
    for i in range(100):
        b1 = torch.randn(16, 8, 32, device='cuda').half()
        o1 = mlp(b1)
        assert o1.dtype == torch.float16
        if threshold > 0:
            assert mlp.fc1.state.idx is not None
        if threshold > 0:
            assert mlp.fc2.state.idx is not None
    assert mlp.fc1.weight.dtype == torch.int8
    assert mlp.fc2.weight.dtype == torch.int8
    mlp = MLP8bit(32, 64, threshold=threshold, has_fp16_weights=False, memory_efficient_backward=memory_efficient_backward).half().to('cuda')
    for i in range(100):
        b1 = torch.randn(16, 8, 32, device='cuda').half()
        o1 = mlp(b1)
        assert o1.dtype == torch.float16
        if threshold > 0:
            assert mlp.fc1.state.idx is not None
        if threshold > 0:
            assert mlp.fc2.state.idx is not None
    assert mlp.fc1.weight.dtype == torch.int8
    assert mlp.fc2.weight.dtype == torch.int8
    assert mlp.fc1.weight.device.type == 'cuda'
    assert mlp.fc2.weight.device.type == 'cuda'
    mlp = MLP8bit(32, 64, threshold=threshold, has_fp16_weights=False, memory_efficient_backward=memory_efficient_backward)
    w1, w2 = (mlp.fc1.weight.clone().cuda(), mlp.fc2.weight.clone().cuda())
    mlp = mlp.cuda().half()
    for i in range(100):
        b1 = torch.randn(16, 8, 32, device='cuda').half()
        o1 = mlp(b1)
        assert o1.dtype == torch.float16
        if threshold > 0:
            assert mlp.fc1.state.idx is not None
        if threshold > 0:
            assert mlp.fc2.state.idx is not None
    assert mlp.fc1.weight.dtype == torch.int8
    assert mlp.fc2.weight.dtype == torch.int8
    assert mlp.fc1.weight.device.type == 'cuda'
    assert mlp.fc2.weight.device.type == 'cuda'
    if memory_efficient_backward:
        b1 = torch.randn(16, 8, 32, device='cuda', requires_grad=True, dtype=torch.half)
        o1 = mlp(b1)
        assert o1.dtype == torch.float16
        assert o1.requires_grad
        grad_proj = torch.randn_like(o1)
        mlp.zero_grad()
        (o1 * grad_proj).sum().backward()
        grad_ref = grad_proj.flatten(2) @ w2.half() @ w1.half()
        scale = grad_ref.abs().mean()
        torch.testing.assert_close(b1.grad, grad_ref, rtol=0, atol=0.05 * scale)
        idx = torch.isclose(b1.grad, grad_ref, atol=0.01 * scale, rtol=0.1)
        assert (idx == 0).sum().item() <= b1.numel() * 0.005