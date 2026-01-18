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
def test_hash_on_str_creation(self):

    def impl(do_hash):
        const1 = 'aaaa'
        const2 = '眼眼眼眼'
        new = const1 + const2
        if do_hash:
            hash(new)
        return new
    jitted = jit(nopython=True)(impl)
    compute_hash = False
    expected = impl(compute_hash)
    got = jitted(compute_hash)
    a = compile_time_get_string_data(expected)
    b = compile_time_get_string_data(got)
    self.assertEqual(a[:-1], b[:-1])
    self.assertTrue(a[-1] != b[-1])
    compute_hash = True
    expected = impl(compute_hash)
    got = jitted(compute_hash)
    a = compile_time_get_string_data(expected)
    b = compile_time_get_string_data(got)
    self.assertEqual(a, b)