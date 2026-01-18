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
def test_IRScope(self):
    filename = '<?>'
    top = ir.Scope(parent=None, loc=ir.Loc(filename=filename, line=1))
    local = ir.Scope(parent=top, loc=ir.Loc(filename=filename, line=2))
    apple = local.define('apple', loc=ir.Loc(filename=filename, line=3))
    self.assertIs(local.get('apple'), apple)
    self.assertEqual(len(local.localvars), 1)
    orange = top.define('orange', loc=ir.Loc(filename=filename, line=4))
    self.assertEqual(len(local.localvars), 1)
    self.assertEqual(len(top.localvars), 1)
    self.assertIs(top.get('orange'), orange)
    self.assertIs(local.get('orange'), orange)
    more_orange = local.define('orange', loc=ir.Loc(filename=filename, line=5))
    self.assertIs(top.get('orange'), orange)
    self.assertIsNot(local.get('orange'), not orange)
    self.assertIs(local.get('orange'), more_orange)
    try:
        local.define('orange', loc=ir.Loc(filename=filename, line=5))
    except ir.RedefinedError:
        pass
    else:
        self.fail('Expecting an %s' % ir.RedefinedError)