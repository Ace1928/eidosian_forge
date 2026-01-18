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
def test_case07_mystery_key_error(self):

    def foo(x):
        with objmode_context():
            t = {'a': x}
            u = 3
        return (x, t, u)
    x = np.array([1, 2, 3])
    cfoo = njit(foo)
    with self.assertRaises(errors.TypingError) as raises:
        cfoo(x)
    exstr = str(raises.exception)
    self.assertIn("Missing type annotation on outgoing variable(s): ['t', 'u']", exstr)
    self.assertIn("Example code: with objmode(t='<add_type_as_string_here>')", exstr)