import itertools
import numpy as np
from numba import jit
from numba.core import utils
from numba.tests.support import TestCase, forbid_codegen
from .enum_usecases import *
import unittest
def test_forbid_codegen(self):
    """
        Test that forbid_codegen() prevents code generation using the @jit
        decorator.
        """

    def f():
        return 1
    with forbid_codegen():
        with self.assertRaises(RuntimeError) as raises:
            cfunc = jit(nopython=True)(f)
            cfunc()
    self.assertIn('codegen forbidden by test case', str(raises.exception))