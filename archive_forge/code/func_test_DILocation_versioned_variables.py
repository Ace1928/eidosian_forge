from collections import namedtuple
import inspect
import re
import numpy as np
import math
from textwrap import dedent
import unittest
import warnings
from numba.tests.support import (TestCase, override_config,
from numba import jit, njit
from numba.core import types
from numba.core.datamodel import default_manager
from numba.core.errors import NumbaDebugInfoWarning
import llvmlite.binding as llvm
def test_DILocation_versioned_variables(self):
    """ Tests that DILocation information for versions of variables matches
        up to their definition site."""

    @njit(debug=True)
    def foo(n):
        if n:
            c = 5
        else:
            c = 1
        py310_defeat1 = 1
        py310_defeat2 = 2
        py310_defeat3 = 3
        py310_defeat4 = 4
        return c
    sig = (types.intp,)
    metadata = self._get_metadata(foo, sig=sig)
    pysrc, pysrc_line_start = inspect.getsourcelines(foo)
    expr = '.*!DILocalVariable\\(name: "c\\$[0-9]?",.*line: ([0-9]+),.*'
    matcher = re.compile(expr)
    associated_lines = set()
    for md in metadata:
        match = matcher.match(md)
        if match:
            groups = match.groups()
            self.assertEqual(len(groups), 1)
            associated_lines.add(int(groups[0]))
    self.assertEqual(len(associated_lines), 2)
    py_lines = set()
    for ix, pyln in enumerate(pysrc):
        if 'c = ' in pyln:
            py_lines.add(ix + pysrc_line_start)
    self.assertEqual(len(py_lines), 2)
    self.assertEqual(associated_lines, py_lines)