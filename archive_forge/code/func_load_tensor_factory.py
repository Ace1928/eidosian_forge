import contextlib
from typing import Sequence
import torch
from torch._custom_op.impl import custom_op
from torch.utils._content_store import ContentStoreReader
@load_tensor.impl_factory()
def load_tensor_factory(name, size, stride, dtype, device):
    if LOAD_TENSOR_READER is None:
        from torch._dynamo.testing import rand_strided
        return rand_strided(size, stride, dtype, device)
    else:
        from torch._dynamo.utils import clone_input
        r = LOAD_TENSOR_READER.read_tensor(name, device=device)
        assert list(r.size()) == size, f'{r.size()} != {size}'
        assert list(r.stride()) == stride, f'{r.stride()} != {stride}'
        assert r.device == device, f'{r.device} != {device}'
        if r.dtype != dtype:
            r = clone_input(r, dtype=dtype)
        return r