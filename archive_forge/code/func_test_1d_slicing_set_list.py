import decimal
import itertools
import numpy as np
import unittest
from numba import jit, njit, typeof
from numba.core import utils, types, errors
from numba.tests.support import TestCase, tag
from numba.core.typing import arraydecl
from numba.core.types import intp, ellipsis, slice2_type, slice3_type
def test_1d_slicing_set_list(self, flags=enable_pyobj_flags):
    """
        List to 1d slice assignment
        """
    self.check_1d_slicing_set_sequence(flags, types.List(types.int16), [8, -42])