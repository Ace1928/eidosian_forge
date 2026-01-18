import decimal
import itertools
import numpy as np
import unittest
from numba import jit, njit, typeof
from numba.core import utils, types, errors
from numba.tests.support import TestCase, tag
from numba.core.typing import arraydecl
from numba.core.types import intp, ellipsis, slice2_type, slice3_type
def slicing_2d_usecase_set(a, b, start, stop, step, start2, stop2, step2):
    a[start:stop:step, start2:stop2:step2] = b
    return a