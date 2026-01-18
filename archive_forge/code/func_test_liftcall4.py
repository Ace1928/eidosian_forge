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
def test_liftcall4(self):
    accept = (errors.TypingError, errors.NumbaRuntimeError, errors.NumbaValueError, errors.CompilerError)
    with self.assertRaises(accept) as raises:
        njit(liftcall4)()
    msg = 'compiler re-entrant to the same function signature'
    self.assertIn(msg, str(raises.exception))