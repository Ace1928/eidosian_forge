import re
import types
import numpy as np
from numba.cuda.testing import unittest, skip_on_cudasim, CUDATestCase
from numba import cuda, jit, float32, int32
from numba.core.errors import TypingError
@skip_on_cudasim('not supported in cudasim')
def test_inspect_llvm(self):

    @cuda.jit(device=True)
    def foo(x, y):
        return x + y
    args = (int32, int32)
    cres = foo.compile_device(args)
    fname = cres.fndesc.mangled_name
    self.assertIn('foo', fname)
    llvm = foo.inspect_llvm(args)
    self.assertIn(fname, llvm)