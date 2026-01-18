from contextlib import contextmanager
import sys
import threading
import traceback
from numba.core import types
import numpy as np
from numba.np import numpy_support
from .vector_types import vector_types
@contextmanager
def swapped_cuda_module(fn, fake_cuda_module):
    from numba import cuda
    fn_globs = fn.__globals__
    orig = dict(((k, v) for k, v in fn_globs.items() if v is cuda))
    repl = dict(((k, fake_cuda_module) for k, v in orig.items()))
    fn_globs.update(repl)
    try:
        yield
    finally:
        fn_globs.update(orig)