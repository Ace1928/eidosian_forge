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
def datetime_samples(self):
    dt_years = ['600', '601', '604', '1968', '1969', '1973', '2000', '2004', '2005', '2100', '2400', '2401']
    dt_suffixes = ['', '-01', '-12', '-02-28', '-12-31', '-01-05T12:30:56Z', '-01-05T12:30:56.008Z']
    dts = [DT(a + b) for a, b in itertools.product(dt_years, dt_suffixes)]
    dts += [DT(s, 'W') for s in dt_years]
    return dts