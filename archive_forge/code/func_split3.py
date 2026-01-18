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
def split3(i):
    """
            Split i's bits into 3 integers.
            """
    i = typ(i)
    return (i & typ(2635249153387078802), i & typ(5270498306774157604), i & typ(10540996613548315209))