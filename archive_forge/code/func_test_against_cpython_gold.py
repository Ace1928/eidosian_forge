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
def test_against_cpython_gold(self):
    args = (('abc', 0, 0), ('abc', 42, 1), ('abcdefghijk', 42, 2), ('äú∑ℇ', 0, 3), ('äú∑ℇ', 42, 4))
    for input_str, seed, position in args:
        with self.subTest(input_str=input_str, seed=seed):
            got = self.get_hash(repr(input_str), seed=seed)
            expected = self.get_expected_hash(position, len(input_str))
            self.assertEqual(got, expected)