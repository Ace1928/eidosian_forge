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
def test_hash_unhashable(self):
    unhashables = (typed.Dict().empty(types.int64, types.int64), typed.List().empty_list(types.int64), np.ones(4))
    cfunc = jit(nopython=True)(hash_usecase)
    for ty in unhashables:
        with self.assertRaises(TypeError) as raises:
            cfunc(ty)
        expected = f"unhashable type: '{str(typeof(ty))}'"
        self.assertIn(expected, str(raises.exception))