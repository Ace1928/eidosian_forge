import inspect
import numpy as np
import triton
import triton.language as tl
from .._C.libtriton.triton import interpreter as _interpreter
def get_ptr_ty(self, elt_ty, addr_space):
    return tl.pointer_type(elt_ty, addr_space)