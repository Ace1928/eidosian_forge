import decimal
import itertools
import numpy as np
import unittest
from numba import jit, njit, typeof
from numba.core import utils, types, errors
from numba.tests.support import TestCase, tag
from numba.core.typing import arraydecl
from numba.core.types import intp, ellipsis, slice2_type, slice3_type
def test_2d_integer_indexing2(self):
    self.test_2d_integer_indexing(pyfunc=integer_indexing_2d_usecase2)
    self.test_2d_integer_indexing(flags=Noflags, pyfunc=integer_indexing_2d_usecase2)