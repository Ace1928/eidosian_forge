import os
import subprocess
import sys
import warnings
import numpy as np
import unittest
from numba import jit
from numba.core.errors import (
from numba.core import errors
from numba.tests.support import ignore_internal_warnings
def test_no_warning_with_forceobj(self):

    def add(x, y):
        a = []
        return x + y
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', NumbaWarning)
        ignore_internal_warnings()
        cfunc = jit(add, forceobj=True)
        cfunc(1, 2)
        self.assertEqual(len(w), 0)