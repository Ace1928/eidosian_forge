import os, sys, subprocess
import dis
import itertools
import numpy as np
import numba
from numba import jit, njit
from numba.core import errors, ir, types, typing, typeinfer, utils
from numba.core.typeconv import Conversion
from numba.extending import overload_method
from numba.tests.support import TestCase, tag
from numba.tests.test_typeconv import CompatibilityTestMixin
from numba.core.untyped_passes import TranslateByteCode, IRProcessing
from numba.core.typed_passes import PartialTypeInference
from numba.core.compiler_machinery import FunctionPass, register_pass
import unittest
@skip_unless_load_fast_and_clear
def test_load_fast_and_clear(self):

    @njit
    def foo(a):
        [x for x in (0,)]
        if a:
            x = 3 + a
        x += 10
        return x
    self.assertEqual(foo(True), foo.py_func(True))
    with self.assertRaises(UnboundLocalError):
        foo.py_func(False)
    self.assertEqual(foo(False), 10)