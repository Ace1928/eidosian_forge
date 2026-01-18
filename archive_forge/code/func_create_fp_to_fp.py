import inspect
import numpy as np
import triton
import triton.language as tl
from .._C.libtriton.triton import interpreter as _interpreter
def create_fp_to_fp(self, src, dst_type):
    assert 'float8 not NotImplemented yet'