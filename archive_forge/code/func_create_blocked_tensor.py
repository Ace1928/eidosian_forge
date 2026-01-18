import inspect
import re
import warnings
from typing import Any, Dict
import torch
from torch.testing import make_tensor
def create_blocked_tensor(B, M, N, blocksize, sparsity, dtype, device):
    assert sparsity <= 1.0 and sparsity >= 0.0, 'sparsity should be a value between 0 and 1'
    assert M % blocksize[0] == 0
    assert N % blocksize[1] == 0
    shape = (B, M // blocksize[0], N // blocksize[1])[int(B == 0):]
    A = torch.bernoulli(torch.full(shape, 1 - sparsity, dtype=dtype, device=device))
    expected_nnz = int((1 - sparsity) * M * N / (blocksize[0] * blocksize[1]))
    nonzero_indices = A.flatten().nonzero()
    actual_nnz = nonzero_indices.shape[0]
    if actual_nnz > expected_nnz:
        selected_nonzeros = torch.randperm(actual_nnz)[:actual_nnz - expected_nnz]
        A.flatten()[nonzero_indices[selected_nonzeros]] = 0
    elif actual_nnz < expected_nnz:
        zero_indices = (A == 0).flatten().nonzero()
        selected_zeros = torch.randperm(zero_indices.shape[0])[:expected_nnz - actual_nnz]
        A.flatten()[zero_indices[selected_zeros]] = 1
    A = torch.repeat_interleave(A, blocksize[0], dim=-2)
    A = torch.repeat_interleave(A, blocksize[1], dim=-1)
    return A