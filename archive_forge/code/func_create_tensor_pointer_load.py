import inspect
import numpy as np
import triton
import triton.language as tl
from .._C.libtriton.triton import interpreter as _interpreter
def create_tensor_pointer_load(self, ptr, boundary_check, padding_option, cache_modifier, eviction_policy, is_volatile):
    ptrs, masks = ptr.materialize_pointers(boundary_check)
    assert padding_option is None
    other = None
    return self.create_masked_load(ptrs, masks, other, cache_modifier, eviction_policy, is_volatile)