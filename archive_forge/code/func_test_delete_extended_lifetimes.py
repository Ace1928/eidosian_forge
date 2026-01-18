import re
from io import StringIO
import numba
from numba.core import types
from numba import jit, njit
from numba.tests.support import override_config, TestCase
import unittest
def test_delete_extended_lifetimes(self):
    sa, sb, cast_ret, dela, delb = self._lifetimes_impl(extend=1)
    self.assertLess(sa, dela)
    self.assertLess(sb, delb)
    self.assertGreater(dela, cast_ret)
    self.assertGreater(delb, cast_ret)