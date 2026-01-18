import itertools
import unittest.mock
from contextlib import contextmanager
from typing import Iterator
import torch
import torch._C
import torch._ops
import torch.utils._python_dispatch
import torch.utils._pytree as pytree
@contextmanager
def enable_crossref_functionalize():
    for op in all_py_loaded_overloads():
        op._uncache_dispatch(torch._C.DispatchKey.Functionalize)
    try:
        with enable_python_dispatcher(), unittest.mock.patch('torch._dispatch.python.CROSSREF_FUNCTIONALIZE', True):
            yield
    finally:
        for op in all_py_loaded_overloads():
            op._uncache_dispatch(torch._C.DispatchKey.Functionalize)