import inspect
import numpy as np
import triton
import triton.language as tl
from .._C.libtriton.triton import interpreter as _interpreter
def create_tensor_pointer_store(self, ptr, value, boundary_check, cache_modifier, eviction_policy):
    ptrs, masks = ptr.materialize_pointers(boundary_check)
    return self.create_masked_store(ptrs, value, masks, cache_modifier, eviction_policy)