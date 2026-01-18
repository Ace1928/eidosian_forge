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
def get_expected_hash(self, position, length):
    if length < sys.hash_info.cutoff:
        algorithm = 'djba33x'
    else:
        algorithm = sys.hash_info.algorithm
    IS_64BIT = not config.IS_32BITS
    if sys.byteorder == 'little':
        platform = 1 if IS_64BIT else 0
    else:
        assert sys.byteorder == 'big'
        platform = 3 if IS_64BIT else 2
    return self.known_hashes[algorithm][position][platform]