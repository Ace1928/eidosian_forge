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
@unittest.skip('Needs hash computation at const unpickling time')
def test_hash_literal(self):

    @jit(nopython=True)
    def fn():
        x = 'abcdefghijklmnopqrstuvwxyz'
        return x
    val = fn()
    tmp = hash('abcdefghijklmnopqrstuvwxyz')
    self.assertEqual(tmp, compile_time_get_string_data(val)[-1])