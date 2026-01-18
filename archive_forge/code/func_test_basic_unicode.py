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
def test_basic_unicode(self):
    kind1_string = 'abcdefghijklmnopqrstuvwxyz'
    for i in range(len(kind1_string)):
        self.check_hash_values([kind1_string[:i]])
    sep = '眼'
    kind2_string = sep.join(list(kind1_string))
    for i in range(len(kind2_string)):
        self.check_hash_values([kind2_string[:i]])
    sep = '🐍⚡'
    kind4_string = sep.join(list(kind1_string))
    for i in range(len(kind4_string)):
        self.check_hash_values([kind4_string[:i]])
    empty_string = ''
    self.check_hash_values(empty_string)