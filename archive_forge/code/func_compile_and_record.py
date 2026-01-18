import collections
import weakref
import gc
import operator
from itertools import takewhile
import unittest
from numba import njit, jit
from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.untyped_passes import PreserveIR
from numba.core.typed_passes import IRLegalization
from numba.core import types, ir
from numba.tests.support import TestCase, override_config, SerialMixin
def compile_and_record(self, pyfunc, raises=None):
    rec = RefRecorder()
    cfunc = self.compile(pyfunc)
    if raises is not None:
        with self.assertRaises(raises):
            cfunc(rec)
    else:
        cfunc(rec)
    return rec