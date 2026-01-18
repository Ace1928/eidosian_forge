import decimal
import itertools
import numpy as np
import unittest
from numba import jit, njit, typeof
from numba.core import utils, types, errors
from numba.tests.support import TestCase, tag
from numba.core.typing import arraydecl
from numba.core.types import intp, ellipsis, slice2_type, slice3_type
def test_1d_slicing7(self, flags=enable_pyobj_flags):
    pyfunc = slicing_1d_usecase7
    self.check_1d_slicing_with_arg(pyfunc, flags)