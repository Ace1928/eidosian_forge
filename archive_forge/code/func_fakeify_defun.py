import itertools
import unittest.mock
from contextlib import contextmanager
from typing import Iterator
import torch
import torch._C
import torch._ops
import torch.utils._python_dispatch
import torch.utils._pytree as pytree
def fakeify_defun(t):
    if isinstance(t, torch.Tensor):
        if torch._is_functional_tensor(t):
            r = torch._from_functional_tensor(t)
            assert t.size() == r.size()
            assert t.stride() == r.stride()
        else:
            r = t
        return fake_mode.from_tensor(r)
    return t