import copy
import warnings
import numpy as np
import numba
from numba.core.transforms import find_setupwiths, with_lifting
from numba.core.withcontexts import bypass_context, call_context, objmode_context
from numba.core.bytecode import FunctionIdentity, ByteCode
from numba.core.interpreter import Interpreter
from numba.core import errors
from numba.core.registry import cpu_target
from numba.core.compiler import compile_ir, DEFAULT_FLAGS
from numba import njit, typeof, objmode, types
from numba.core.extending import overload
from numba.tests.support import (MemoryLeak, TestCase, captured_stdout,
from numba.core.utils import PYVERSION
from numba.experimental import jitclass
import unittest
@expected_failure_for_list_arg
def test_case01_mutate_list_ahead_of_ctx(self):

    def foo(x, z):
        x[2] = z
        with objmode_context():
            print(x)
        with objmode_context():
            x[2] = 2 * z
            print(x)
        return x
    self.assert_equal_return_and_stdout(foo, [1, 2, 3], 15)