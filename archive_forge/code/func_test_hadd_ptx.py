import itertools
import numpy as np
import operator
import re
from numba import cuda, int64
from numba.cuda import compile_ptx
from numba.core.errors import TypingError
from numba.core.types import f2
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
@skip_on_cudasim('Compilation unsupported in the simulator')
def test_hadd_ptx(self):
    args = (f2[:], f2, f2)
    ptx, _ = compile_ptx(simple_hadd_scalar, args, cc=(5, 3))
    self.assertIn('add.f16', ptx)