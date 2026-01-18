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
def test_boxer_raise(self):

    @unbox(self.DynTypeType)
    def unboxer(typ, obj, c):
        return NativeValue(c.context.get_dummy_value())

    @box(self.DynTypeType)
    def boxer(typ, val, c):

        def bridge(x):
            if x > 0:
                raise ValueError('cannot do x > 0')
            return x
        args = [c.context.get_constant(types.intp, 1)]
        sig = signature(types.intp, types.intp)
        is_error, res = c.pyapi.call_jit_code(bridge, sig, args)
        retval = cgutils.alloca_once(c.builder, c.pyapi.pyobj, zfill=True)
        with c.builder.if_then(c.builder.not_(is_error)):
            obj = c.pyapi.long_from_ssize_t(res)
            c.builder.store(obj, retval)
        return c.builder.load(retval)

    @njit
    def passthru(x):
        return x
    with self.assertRaises(ValueError) as raises:
        passthru(self.dyn_type)
    self.assertIn('cannot do x > 0', str(raises.exception))