from typing import Tuple
import pytest
import torch
import bitsandbytes as bnb
from tests.helpers import (
@pytest.mark.parametrize('dim1', get_test_dims(16, 64, n=1), ids=id_formatter('dim1'))
@pytest.mark.parametrize('dim2', [*get_test_dims(32, 96, n=1), 0], ids=id_formatter('dim2'))
@pytest.mark.parametrize('dim3', get_test_dims(32, 96, n=1), ids=id_formatter('dim3'))
@pytest.mark.parametrize('dim4', get_test_dims(32, 96, n=1), ids=id_formatter('dim4'))
@pytest.mark.parametrize('funcs', [(torch.matmul, bnb.matmul_4bit)], ids=['func=matmul'])
@pytest.mark.parametrize('req_grad', BOOLEAN_TRIPLES, ids=id_formatter('req_grad'))
@pytest.mark.parametrize('transpose', TRANSPOSE_VALS, ids=id_formatter('transpose'))
@pytest.mark.parametrize('has_bias', TRUE_FALSE, ids=id_formatter('has_bias'))
@pytest.mark.parametrize('dtype', [torch.float16, torch.float32], ids=describe_dtype)
@pytest.mark.parametrize('compress_statistics', TRUE_FALSE, ids=id_formatter('compress_statistics'))
@pytest.mark.parametrize('quant_type', ['fp4', 'nf4'], ids=id_formatter('quant_type'))
def test_matmul_4bit(dim1, dim2, dim3, dim4, funcs, dtype, req_grad, transpose, has_bias, compress_statistics, quant_type):
    dimA = (dim2, dim3) if not transpose[0] else (dim3, dim2)
    dimB = (dim3, dim4) if not transpose[1] else (dim4, dim3)
    if has_bias == False:
        req_grad = list(req_grad)
        req_grad[2] = False
    for i in range(3):
        if funcs[0] in [torch.mm, torch.matmul]:
            A = torch.randn(size=dimA, device='cuda', requires_grad=req_grad[0], dtype=dtype)
            B = torch.randn(size=dimB, device='cuda', requires_grad=req_grad[1], dtype=dtype)
            target = torch.randn(size=(dim2, dim4), device='cuda', requires_grad=req_grad[1], dtype=dtype)
            bias = None
            bias2 = None
            if has_bias:
                bias = torch.randn(dim4, device='cuda', dtype=dtype, requires_grad=req_grad[2])
                bias2 = bias.clone()
            torch.nn.init.xavier_uniform_(B)
            B2, quant_state = bnb.functional.quantize_4bit(B, compress_statistics=compress_statistics, quant_type=quant_type)
            if not transpose[0] and transpose[1]:
                out_torch = funcs[0](A, B.t())
                out_bnb = funcs[1](A, B2.t(), quant_state, bias=bias2)
            elif not transpose[0] and (not transpose[1]):
                out_torch = funcs[0](A, B)
                out_bnb = funcs[1](A, B2, quant_state, bias=bias2)
            if has_bias:
                out_torch += bias
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
                if has_bias:
                    gradBias1 = bias.grad
                    bias.grad = None
                loss_torch = torch.nn.functional.mse_loss(out_torch, target).mean()
                loss_torch.backward()
                gradA2 = A.grad
                gradB2 = B.grad
                A.grad = None
                B.grad = None
                if has_bias:
                    gradBias2 = bias.grad
                    bias.grad = None
                if req_grad[0]:
                    torch.testing.assert_close(gradA1, gradA2, atol=0.015, rtol=0.1)
                if req_grad[2]:
                    torch.testing.assert_close(gradBias1, gradBias2)