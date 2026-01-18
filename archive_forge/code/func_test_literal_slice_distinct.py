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
def test_literal_slice_distinct(self):
    sl1 = types.misc.SliceLiteral(slice(1, None, None))
    sl2 = types.misc.SliceLiteral(slice(None, None, None))
    sl3 = types.misc.SliceLiteral(slice(1, None, None))
    self.assertNotEqual(sl1, sl2)
    self.assertEqual(sl1, sl3)