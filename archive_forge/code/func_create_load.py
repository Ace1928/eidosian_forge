import inspect
import numpy as np
import triton
import triton.language as tl
from .._C.libtriton.triton import interpreter as _interpreter
def create_load(self, ptr, _0, _1, is_volatile):
    mask = TensorHandle(np.ones_like(ptr.data, dtype=bool), tl.int1)
    other = None
    return self.create_masked_load(ptr, mask, other, _0, _1, is_volatile)