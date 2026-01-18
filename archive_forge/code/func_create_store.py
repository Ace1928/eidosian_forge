import inspect
import numpy as np
import triton
import triton.language as tl
from .._C.libtriton.triton import interpreter as _interpreter
def create_store(self, ptr, val, _0, _1):
    mask = TensorHandle(np.ones_like(ptr.data, dtype=bool), tl.int1)
    return self.create_masked_store(ptr, val, mask, None, None)