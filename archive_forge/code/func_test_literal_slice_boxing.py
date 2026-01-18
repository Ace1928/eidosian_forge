from functools import partial
import itertools
from itertools import chain, product, starmap
import sys
import numpy as np
from numba import jit, literally, njit, typeof, TypingError
from numba.core import utils, types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.types.functions import _header_lead
import unittest
def test_literal_slice_boxing(self):

    @njit
    def f(x):
        return literally(x)
    slices = (slice(1, 4, 2), slice(1, 2), slice(1), slice(None, 1, 1), slice(1, None, 1), slice(None, None, 1), slice(None), slice(None, None, None))
    for sl in slices:
        self.assertEqual(sl, f(sl))