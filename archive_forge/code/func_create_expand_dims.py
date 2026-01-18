import inspect
import numpy as np
import triton
import triton.language as tl
from .._C.libtriton.triton import interpreter as _interpreter
def create_expand_dims(self, arg, axis):
    return TensorHandle(np.expand_dims(arg.data, axis), arg.dtype)