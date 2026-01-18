import contextlib
import itertools
import re
import unittest
import warnings
import numpy as np
from numba import jit, vectorize, njit
from numba.np.numpy_support import numpy_version
from numba.core import types, config
from numba.core.errors import TypingError
from numba.tests.support import TestCase, tag, skip_parfors_unsupported
from numba.np import npdatetime_helpers, numpy_support
def test_hash(self):
    f = self.jit(hash_usecase)

    def check(a):
        self.assertPreciseEqual(f(a), hash(a))
    TD_CASES = ((3,), (-4,), (3, 'ms'), (-4, 'ms'), (27, 'D'), (2, 'D'), (2, 'W'), (2, 'Y'), (3, 'W'), (365, 'D'), (10000, 'D'), (-10000, 'D'), ('NaT',), ('NaT', 'ms'), ('NaT', 'D'), (-1,))
    DT_CASES = (('2014',), ('2016',), ('2000',), ('2014-02',), ('2014-03',), ('2014-04',), ('2016-02',), ('2000-12-31',), ('2014-01-16',), ('2014-01-05',), ('2014-01-07',), ('2014-01-06',), ('2014-02-02',), ('2014-02-27',), ('2014-02-16',), ('2014-03-01',), ('2000-01-01T01:02:03.002Z',), ('2000-01-01T01:02:03Z',), ('NaT',))
    for case, typ in zip(TD_CASES + DT_CASES, (TD,) * len(TD_CASES) + (DT,) * len(TD_CASES)):
        check(typ(*case))