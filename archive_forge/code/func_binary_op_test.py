import functools
import itertools
import sys
import warnings
import threading
import operator
import numpy as np
import unittest
from numba import guvectorize, njit, typeof, vectorize
from numba.core import types
from numba.np.numpy_support import from_dtype
from numba.core.errors import LoweringError, TypingError
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.typing.npydecl import supported_ufuncs
from numba.np import numpy_support
from numba.core.registry import cpu_target
from numba.core.base import BaseContext
from numba.np import ufunc_db
def binary_op_test(self, operator, nrt=True, skip_inputs=[], additional_inputs=[], int_output_type=None, float_output_type=None, positive_rhs=False):
    operator_func = _make_binary_ufunc_op_usecase(operator)
    inputs = list(self.inputs)
    inputs.extend(additional_inputs)
    pyfunc = operator_func
    random_state = np.random.RandomState(1)
    for input_tuple in inputs:
        input_operand1, input_type = input_tuple
        input_dtype = numpy_support.as_dtype(getattr(input_type, 'dtype', input_type))
        input_type1 = input_type
        if input_type in skip_inputs:
            continue
        if positive_rhs:
            zero = np.zeros(1, dtype=input_dtype)[0]
        if isinstance(input_type, types.Array):
            input_operand0 = input_operand1
            input_type0 = input_type
            if positive_rhs and np.any(input_operand1 < zero):
                continue
        else:
            input_operand0 = random_state.uniform(0, 100, 10).astype(input_dtype)
            input_type0 = typeof(input_operand0)
            if positive_rhs and input_operand1 < zero:
                continue
        args = (input_type0, input_type1)
        cfunc = self._compile(pyfunc, args, nrt=nrt)
        expected = pyfunc(input_operand0, input_operand1)
        got = cfunc(input_operand0, input_operand1)
        self._check_results(expected, got)