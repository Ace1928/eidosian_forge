from typing import Tuple
import pytest
import torch
import bitsandbytes as bnb
from tests.helpers import (
@pytest.mark.parametrize('dim1', get_test_dims(16, 64, n=1), ids=id_formatter('dim1'))
@pytest.mark.parametrize('dim2', [*get_test_dims(32, 96, n=1), 0], ids=id_formatter('dim2'))
@pytest.mark.parametrize('dim3', get_test_dims(32, 96, n=1), ids=id_formatter('dim3'))
@pytest.mark.parametrize('dim4', get_test_dims(32, 96, n=1), ids=id_formatter('dim4'))
@pytest.mark.parametrize('req_grad', BOOLEAN_TRIPLES, ids=id_formatter('req_grad'))
@pytest.mark.parametrize('transpose', TRANSPOSE_VALS, ids=id_formatter('transpose'))
@pytest.mark.parametrize('dtype', [torch.float16, torch.float32], ids=describe_dtype)
@pytest.mark.parametrize('funcs', [(torch.matmul, bnb.research.matmul_fp8_mixed), (torch.matmul, bnb.research.matmul_fp8_global)], ids=['matmul_fp8_mixed', 'matmul_fp8_global'])
def test_matmul_fp8(dim1, dim2, dim3, dim4, funcs, dtype, req_grad, transpose):
    dimA = (dim2, dim3) if not transpose[0] else (dim3, dim2)
    dimB = (dim3, dim4) if not transpose[1] else (dim4, dim3)
    req_grad = list(req_grad)
    req_grad[2] = False
    for i in range(3):
        if funcs[0] in [torch.mm, torch.matmul]:
            A = torch.randn(size=dimA, device='cuda', requires_grad=req_grad[0], dtype=dtype)
            B = torch.randn(size=dimB, device='cuda', requires_grad=req_grad[1], dtype=dtype)
            target = torch.randn(size=(dim2, dim4), device='cuda', requires_grad=req_grad[1], dtype=dtype)
            torch.nn.init.xavier_uniform_(B)
            fw_code = bnb.functional.create_fp8_map(True, 4, 3, 8).to(A.device)
            bw_code = bnb.functional.create_fp8_map(True, 5, 2, 8).to(A.device)
            if not transpose[0] and transpose[1]:
                out_torch = funcs[0](A, B.t())
                out_bnb = funcs[1](A, B.t(), fw_code, bw_code)
            elif not transpose[0] and (not transpose[1]):
                out_torch = funcs[0](A, B)
                out_bnb = funcs[1](A, B, fw_code, bw_code)
            assert out_bnb.dtype == A.dtype, f'bnb matmullt received {A.dtype} but returned {out_bnb.dtype}'
            n = out_bnb.numel()
            err = torch.abs(out_bnb - out_torch).float().mean().item()
            if n > 0:
                assert err < 0.115
            if any(req_grad):
                out_bnb.data.copy_(out_torch)
                torch.cuda.synchronize()
                loss_bnb = torch.nn.functional.mse_loss(out_bnb, target).mean()
                loss_bnb.backward()
                gradA1 = A.grad
                gradB1 = B.grad
                A.grad = None
                B.grad = None
                loss_torch = torch.nn.functional.mse_loss(out_torch, target).mean()
                loss_torch.backward()
                gradA2 = A.grad
                gradB2 = B.grad
                A.grad = None
                B.grad = None
                if req_grad[0]:
                    torch.testing.assert_close(gradA1, gradA2, atol=0.015, rtol=0.1)
                if req_grad[1]:
                    n = gradB1.numel()
                    if dim2 > 0:
                        assert torch.abs(gradB1).sum() > 0.0
                        assert torch.abs(gradB2).sum() > 0.0
                    else:
                        assert torch.abs(gradB1).sum() == 0.0
                        assert torch.abs(gradB2).sum() == 0.0
                    idx = torch.isclose(gradB1, gradB2, atol=0.06, rtol=0.3)
                    assert (idx == 0).sum().item() <= n * 0.1
                    idx = torch.isclose(gradB1, gradB2, atol=0.1, rtol=0.3)
                    assert (idx == 0).sum().item() <= n * 0.02
                    grad_err = (gradB1 - gradB2).abs().mean()
                    assert grad_err.item() < 0.003
                    torch.testing.assert_close(gradB1, gradB2, atol=0.18, rtol=0.3)