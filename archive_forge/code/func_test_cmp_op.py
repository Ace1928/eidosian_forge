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
def test_cmp_op(self):
    """
        Verify that comparison operators are supported by the rewriter.
        """
    ns = self._test_root_function(are_roots_imaginary)
    self._assert_total_rewrite(ns.control_pipeline.state.func_ir.blocks, ns.test_pipeline.state.func_ir.blocks)