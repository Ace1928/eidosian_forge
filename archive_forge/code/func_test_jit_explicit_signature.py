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
def test_jit_explicit_signature(self):

    def _check_explicit_signature(sig):
        f = jit(sig, nopython=True)(add_usecase)
        args = (DT(1, 'ms'), TD(2, 'us'))
        expected = add_usecase(*args)
        self.assertPreciseEqual(f(*args), expected)
    sig = types.NPDatetime('us')(types.NPDatetime('ms'), types.NPTimedelta('us'))
    _check_explicit_signature(sig)
    sig = "NPDatetime('us')(NPDatetime('ms'), NPTimedelta('us'))"
    _check_explicit_signature(sig)