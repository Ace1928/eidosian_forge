import inspect
import math
import operator
import sys
import pickle
import multiprocessing
import ctypes
import warnings
import re
import numpy as np
from llvmlite import ir
import numba
from numba import njit, jit, vectorize, guvectorize, objmode
from numba.core import types, errors, typing, compiler, cgutils
from numba.core.typed_passes import type_inference_stage
from numba.core.registry import cpu_target
from numba.core.imputils import lower_constant
from numba.tests.support import (
from numba.core.errors import LoweringError
import unittest
from numba.extending import (
from numba.core.typing.templates import (
from .pdlike_usecase import Index, Series
def test_deserialization(self):
    """
        Test deserialization of intrinsic
        """

    def defn(context, x):

        def codegen(context, builder, signature, args):
            return args[0]
        return (x(x), codegen)
    memo = _Intrinsic._memo
    memo_size = len(memo)
    original = _Intrinsic('foo', defn)
    self.assertIs(original._defn, defn)
    pickled = pickle.dumps(original)
    memo_size += 1
    self.assertEqual(memo_size, len(memo))
    del original
    self.assertEqual(memo_size, len(memo))
    _Intrinsic._recent.clear()
    memo_size -= 1
    self.assertEqual(memo_size, len(memo))
    rebuilt = pickle.loads(pickled)
    self.assertIsNot(rebuilt._defn, defn)
    second = pickle.loads(pickled)
    self.assertIs(rebuilt._defn, second._defn)