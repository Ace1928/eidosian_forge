import inspect
import numpy as np
import triton
import triton.language as tl
from .._C.libtriton.triton import interpreter as _interpreter
def create_broadcast(self, arg, shape):
    return TensorHandle(np.broadcast_to(arg.data, shape), arg.dtype)