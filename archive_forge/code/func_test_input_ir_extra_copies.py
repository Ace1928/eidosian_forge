from numba import jit, njit
from numba.core import types, ir, config, compiler
from numba.core.registry import cpu_target
from numba.core.annotations import type_annotations
from numba.core.ir_utils import (copy_propagate, apply_copy_propagate,
from numba.core.typed_passes import type_inference_stage
from numba.tests.support import IRPreservingTestPipeline
import numpy as np
import unittest
def test_input_ir_extra_copies(self):
    """make sure Interpreter._remove_unused_temporaries() has removed extra copies
        in the IR in simple cases so copy propagation is faster
        """

    def test_impl(a):
        b = a + 3
        return b
    j_func = njit(pipeline_class=IRPreservingTestPipeline)(test_impl)
    self.assertEqual(test_impl(5), j_func(5))
    fir = j_func.overloads[j_func.signatures[0]].metadata['preserved_ir']
    self.assertTrue(len(fir.blocks) == 1)
    block = next(iter(fir.blocks.values()))
    b_found = False
    for stmt in block.body:
        if isinstance(stmt, ir.Assign) and stmt.target.name == 'b':
            b_found = True
            self.assertTrue(isinstance(stmt.value, ir.Expr) and stmt.value.op == 'binop' and (stmt.value.lhs.name == 'a'))
    self.assertTrue(b_found)