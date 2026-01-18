import gc
from io import StringIO
import numpy as np
from numba import njit, vectorize
from numba import typeof
from numba.core import utils, types, typing, ir, compiler, cpu, cgutils
from numba.core.compiler import Compiler, Flags
from numba.core.registry import cpu_target
from numba.tests.support import (MemoryLeakMixin, TestCase, temp_directory,
from numba.extending import (
import operator
import textwrap
import unittest
def test_complicated_expr(self):
    """
        Using the polynomial root function, ensure the full expression is
        being put in the same kernel with no remnants of intermediate
        array expressions.
        """
    ns = self._test_root_function()
    self._assert_total_rewrite(ns.control_pipeline.state.func_ir.blocks, ns.test_pipeline.state.func_ir.blocks)