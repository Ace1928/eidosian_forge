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
def test_heterogeneous_tuples(self):
    modulo = 2 ** 63

    def split(i):
        a = i & 6148914691236517205
        b = i & 2863311530 ^ i >> 32 & 2863311530
        return (np.int64(a), np.float64(b * 0.0001))
    self.check_tuples(self.int_samples(), split)