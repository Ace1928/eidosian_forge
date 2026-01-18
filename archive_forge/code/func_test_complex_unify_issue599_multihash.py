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
def test_complex_unify_issue599_multihash(self):
    """
        Test issue #599 for multiple values of PYTHONHASHSEED.
        """
    env = os.environ.copy()
    for seedval in (1, 2, 1024):
        env['PYTHONHASHSEED'] = str(seedval)
        subproc = subprocess.Popen([sys.executable, '-c', 'import numba.tests.test_typeinfer as test_mod\n' + 'test_mod.TestUnifyUseCases._actually_test_complex_unify()'], env=env)
        subproc.wait()
        self.assertEqual(subproc.returncode, 0, 'Child process failed.')