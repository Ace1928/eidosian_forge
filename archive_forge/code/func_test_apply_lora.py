import pytest
import torch
from vllm.lora.layers import _apply_lora, _apply_lora_packed_nslice
from .utils import DummyLoRAManager
@pytest.mark.parametrize('m', TENSOR_SIZES)
@pytest.mark.parametrize('n', TENSOR_SIZES)
@pytest.mark.parametrize('k', BATCH_SIZES)
@pytest.mark.parametrize('rank', RANKS)
@pytest.mark.parametrize('dtype', DTYPES)
def test_apply_lora(m, n, k, rank, dtype) -> None:
    manager = DummyLoRAManager()
    module_name = 'module'
    weight = torch.rand([m, n], device='cuda', dtype=dtype)
    manager.init_random_lora(module_name, weight, rank=rank)
    lora = manager.get_module_lora(module_name)
    input = torch.rand(k, n, device='cuda', dtype=dtype)
    expected = input @ lora.lora_a @ lora.lora_b * lora.scaling
    lora_a_stack = torch.zeros(8, 1, lora.lora_a.shape[1], lora.lora_a.shape[0], device='cuda', dtype=dtype)
    lora_b_stack = torch.zeros(8, 1, lora.lora_b.shape[1], lora.lora_b.shape[0], device='cuda', dtype=dtype)
    for i in range(lora_a_stack.shape[0]):
        lora_a_stack[i][0] = lora.lora_a.T
        lora_b_stack[i][0] = (lora.lora_b * lora.scaling).T
    output = torch.zeros(k, m, device='cuda', dtype=dtype)
    _apply_lora(input, lora_a_stack, lora_b_stack, torch.randint(0, lora_a_stack.shape[0], (len(input),), device='cuda'), output)
    rtol, atol = TOLERANCES[dtype]
    assert torch.allclose(expected, output, rtol=rtol, atol=atol)
    output[:] = 0
    _apply_lora(input, lora_a_stack, lora_b_stack, torch.full((len(input),), -1, device='cuda'), output)
    assert torch.allclose(torch.zeros_like(output), output)
    manager.reset_lora()