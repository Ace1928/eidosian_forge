import inspect
import numpy as np
import triton
import triton.language as tl
from .._C.libtriton.triton import interpreter as _interpreter
def create_advance(self, ptr, offsets):
    assert len(ptr.offsets) == len(offsets)
    ret = BlockPointerHandle(ptr.base, ptr.shape, ptr.strides, ptr.offsets, ptr.tensor_shape, ptr.order)
    for i in range(len(offsets)):
        ret.offsets[i].data += offsets[i].data
    return ret