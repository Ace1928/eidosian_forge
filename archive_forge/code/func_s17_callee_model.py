import operator
import warnings
from itertools import product
import numpy as np
from numba import njit, typeof, literally, prange
from numba.core import types, ir, ir_utils, cgutils, errors, utils
from numba.core.extending import (
from numba.core.cpu import InlineOptions
from numba.core.compiler import DefaultPassBuilder, CompilerBase
from numba.core.typed_passes import InlineOverloads
from numba.core.typing import signature
from numba.tests.support import (TestCase, unittest,
def s17_callee_model(expr, caller_info, callee_info):
    self.assertIsInstance(expr, ir.Expr)
    self.assertEqual(expr.op, 'call')
    return self.sentinel_17_cost_model(callee_info.func_ir)