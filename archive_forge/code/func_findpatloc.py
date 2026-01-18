import re
from io import StringIO
import numba
from numba.core import types
from numba import jit, njit
from numba.tests.support import override_config, TestCase
import unittest
def findpatloc(self, lines, pat):
    for i, ln in enumerate(lines):
        if pat in ln:
            return i
    raise ValueError("can't find {!r}".format(pat))