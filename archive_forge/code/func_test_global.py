import unittest
from unittest.case import TestCase
import warnings
import numpy as np
from numba import objmode
from numba.core import ir, compiler
from numba.core import errors
from numba.core.compiler import (
from numba.core.compiler_machinery import (
from numba.core.untyped_passes import (
from numba import njit
def test_global(self):
    a = ir.Global('foo', 0, self.loc1)
    b = ir.Global('foo', 0, self.loc1)
    c = ir.Global('foo', 0, self.loc2)
    d = ir.Global('bar', 0, self.loc1)
    e = ir.Global('foo', 1, self.loc1)
    self.check(a, same=[b, c], different=[d, e])