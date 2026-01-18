import inspect
import numpy as np
import triton
import triton.language as tl
from .._C.libtriton.triton import interpreter as _interpreter
def create_make_range(self, start, stop):
    return TensorHandle(np.arange(start, stop, dtype=np.int32), tl.int32)