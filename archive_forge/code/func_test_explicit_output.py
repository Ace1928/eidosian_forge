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
def test_explicit_output(self):
    """
        Check that ufunc calls with explicit outputs are not rewritten.
        """
    ns = self._test_explicit_output_function(explicit_output)
    self._assert_no_rewrite(ns.control_pipeline.state.func_ir.blocks, ns.test_pipeline.state.func_ir.blocks)