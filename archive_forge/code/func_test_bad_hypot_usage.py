import math
import re
import textwrap
import operator
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.core.errors import TypingError
from numba.core.types.functions import _header_lead
from numba.tests.support import TestCase
def test_bad_hypot_usage(self):
    with self.assertRaises(TypingError) as raises:
        njit(())(bad_hypot_usage)
    errmsg = str(raises.exception)
    self.assertIn(' * (float64, float64) -> float64', errmsg)
    ctx_lines = [x for x in errmsg.splitlines() if 'During:' in x]
    self.assertTrue(re.search('.*During: resolving callee type: Function.*hypot', ctx_lines[0]))
    self.assertTrue(re.search('.*During: typing of call .*test_typingerror.py', ctx_lines[1]))