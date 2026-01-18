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
def check_extracted_with(self, func, expect_count, expected_stdout):
    the_ir = get_func_ir(func)
    new_ir, extracted = with_lifting(the_ir, self.typingctx, self.targetctx, self.flags, locals={})
    self.assertEqual(len(extracted), expect_count)
    cres = self.compile_ir(new_ir)
    with captured_stdout() as out:
        cres.entry_point()
    self.assertEqual(out.getvalue(), expected_stdout)