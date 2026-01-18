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
def test_loc(self):
    a = ir.Loc('file', 1, 0)
    b = ir.Loc('file', 1, 0)
    c = ir.Loc('pile', 1, 0)
    d = ir.Loc('file', 2, 0)
    e = ir.Loc('file', 1, 1)
    self.check(a, same=[b], different=[c, d, e])
    f = ir.Loc('file', 1, 0, maybe_decorator=False)
    g = ir.Loc('file', 1, 0, maybe_decorator=True)
    self.check(a, same=[f, g])