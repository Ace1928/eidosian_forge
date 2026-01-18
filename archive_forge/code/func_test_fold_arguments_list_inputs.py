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
def test_fold_arguments_list_inputs(self):
    cases = [dict(func=lambda a, b, c, d: None, args=['arg.a', 'arg.b'], kws=dict(c='arg.c', d='arg.d')), dict(func=lambda: None, args=[], kws=dict()), dict(func=lambda a: None, args=['arg.a'], kws={}), dict(func=lambda a: None, args=[], kws=dict(a='arg.a'))]
    for case in cases:
        with self.subTest(**case):
            self.check_fold_arguments_list_inputs(**case)