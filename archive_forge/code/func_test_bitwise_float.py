import copy
import itertools
import operator
import unittest
import numpy as np
from numba import jit, njit
from numba.core import types, utils, errors
from numba.core.types.functions import _header_lead
from numba.tests.support import TestCase, tag, needs_blas
from numba.tests.matmul_usecase import (matmul_usecase, imatmul_usecase,
def test_bitwise_float(self):
    """
        Make sure that bitwise float operations are not allowed
        """

    def assert_reject_compile(pyfunc, argtypes, opname):
        msg = 'expecting TypingError when compiling {}'.format(pyfunc)
        with self.assertRaises(errors.TypingError, msg=msg) as raises:
            njit(argtypes)(pyfunc)
        fmt = _header_lead + ' {}'
        expecting = fmt.format(opname if isinstance(opname, str) else 'Function({})'.format(opname))
        self.assertIn(expecting, str(raises.exception))
    methods = ['bitshift_left_usecase', 'bitshift_ileft_usecase', 'bitshift_right_usecase', 'bitshift_iright_usecase', 'bitwise_and_usecase', 'bitwise_iand_usecase', 'bitwise_or_usecase', 'bitwise_ior_usecase', 'bitwise_xor_usecase', 'bitwise_ixor_usecase', 'bitwise_not_usecase_binary']
    for name in methods:
        pyfunc = getattr(self.op, name)
        assert_reject_compile(pyfunc, (types.float32, types.float32), opname=self._bitwise_opnames[name])