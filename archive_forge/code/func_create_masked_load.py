import inspect
import numpy as np
import triton
import triton.language as tl
from .._C.libtriton.triton import interpreter as _interpreter
def create_masked_load(self, ptrs, mask, other, cache_modifier, eviction_policy, is_volatile):
    dtype_tt = ptrs.dtype.element_ty
    dtype_np = self.np_dtype(dtype_tt)
    if other is None:
        other = TensorHandle(np.ones_like(ptrs.data, dtype=dtype_np), dtype_tt)
    ret = _interpreter.load(ptrs.data, mask.data, other.data, dtype_np)
    return TensorHandle(ret, dtype_tt)