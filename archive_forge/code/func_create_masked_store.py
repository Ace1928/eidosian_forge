import inspect
import numpy as np
import triton
import triton.language as tl
from .._C.libtriton.triton import interpreter as _interpreter
def create_masked_store(self, ptrs, value, mask, cache_modifier, eviction_policy):
    return _interpreter.store(ptrs.data, value.data, mask.data)