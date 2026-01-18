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
def test_load_fast_and_clear_variant_2(self):

    @njit
    def foo():
        if False:
            x = 1
        [x for x in (1,)]
        return x
    with self.assertRaises(errors.TypingError) as raises:
        foo()
    self.assertIn('return value is undefined', str(raises.exception))