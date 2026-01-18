import unittest
import os
import sys
import subprocess
from collections import defaultdict
from textwrap import dedent
import numpy as np
from numba import jit, config, typed, typeof
from numba.core import types, utils
import unittest
from numba.tests.support import (TestCase, skip_unless_py10_or_later,
from numba.cpython.unicode import compile_time_get_string_data
from numba.cpython import hashing
def complex_samples(self, typ, float_ty):
    for real in self.float_samples(float_ty):
        for imag in self.float_samples(float_ty):
            real = real[:len(imag)]
            imag = imag[:len(real)]
            a = real + typ(1j) * imag
            if utils.PYVERSION >= (3, 10):
                if not np.any(np.isnan(a)):
                    yield a
            else:
                yield a