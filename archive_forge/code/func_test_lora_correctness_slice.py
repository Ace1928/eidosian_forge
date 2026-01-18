import pytest
import torch
import vllm.lora.punica as punica
@pytest.mark.parametrize('dtype_str', ['float16', 'bfloat16'])
@pytest.mark.parametrize('h1', H1)
@pytest.mark.parametrize('h2', H2)
@pytest.mark.parametrize('seed', SEED)
@torch.inference_mode()
def test_lora_correctness_slice(dtype_str, h1, h2, seed):
    if h2 % 3 != 0 or h2 // 3 not in H1:
        pytest.skip('h2 must be divisible by 3 and in supported shapes')
    torch.manual_seed(seed)
    num_loras = 4
    num_layers = 1
    r = 8
    bs = 32
    scale = 0.123
    dtype = getattr(torch, dtype_str)
    device = torch.device('cuda')
    wa_T_all_0 = torch.randn(num_loras, num_layers, r, h1, dtype=dtype, device=device)
    wa_T_all_1 = torch.randn(num_loras, num_layers, r, h1, dtype=dtype, device=device)
    wa_T_all_2 = torch.randn(num_loras, num_layers, r, h1, dtype=dtype, device=device)
    wb_T_all_0 = torch.randn(num_loras, num_layers, h2 // 3, r, dtype=dtype, device=device)
    wb_T_all_1 = torch.randn(num_loras, num_layers, h2 // 3, r, dtype=dtype, device=device)
    wb_T_all_2 = torch.randn(num_loras, num_layers, h2 // 3, r, dtype=dtype, device=device)
    indices = torch.randint(num_loras, (bs,), dtype=torch.long, device=device)
    for layer_idx in range(num_layers):
        x = torch.randn(bs, h1, dtype=dtype, device=device)
        y = torch.randn(bs, h2, dtype=dtype, device=device)
        s = h2 // 3
        y_ref = y.clone()
        _lora_ref_impl(y_ref[:, :s], x, wa_T_all_0, wb_T_all_0, indices, layer_idx, scale)
        _lora_ref_impl(y_ref[:, s:s * 2], x, wa_T_all_1, wb_T_all_1, indices, layer_idx, scale)
        _lora_ref_impl(y_ref[:, s * 2:], x, wa_T_all_2, wb_T_all_2, indices, layer_idx, scale)
        y_our = y.clone()
        punica.add_lora_slice(y_our, x, wa_T_all_0, wb_T_all_0, indices, layer_idx, scale, 0, s)
        punica.add_lora_slice(y_our, x, wa_T_all_1, wb_T_all_1, indices, layer_idx, scale, s, s)
        punica.add_lora_slice(y_our, x, wa_T_all_2, wb_T_all_2, indices, layer_idx, scale, s * 2, s)
        assert_close(y_ref[:, :s], y_our[:, :s])
        assert_close(y_ref[:, s:s * 2], y_our[:, s:s * 2])
        assert_close(y_ref[:, s * 2:], y_our[:, s * 2:])