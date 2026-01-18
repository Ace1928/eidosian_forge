from collections import namedtuple
import numpy as np
from numba.tests.support import (TestCase, MemoryLeakMixin,
from numba import njit, typed, literal_unroll, prange
from numba.core import types, errors, ir
from numba.testing import unittest
from numba.core.extending import overload
from numba.core.compiler_machinery import (PassManager, register_pass,
from numba.core.compiler import CompilerBase
from numba.core.untyped_passes import (FixupArgs, TranslateByteCode,
from numba.core.typed_passes import (NopythonTypeInference, IRLegalization,
from numba.core.ir_utils import (compute_cfg_from_blocks, flatten_labels)
from numba.core.types.functions import _header_lead
def test_inlined_loops(self):
    """ Checks a loop appearing from a closure """

    def get_info(pipeline):

        @njit(pipeline_class=pipeline)
        def foo(tup):

            def bar(n):
                acc = 0
                for i in range(n):
                    acc += 1
                return acc
            acc = 0
            for i in tup:
                acc += i
                acc += bar(i)
            return acc
        x = (1, 2, 3)
        self.assertEqual(foo(x), foo.py_func(x))
        cres = foo.overloads[foo.signatures[0]]
        func_ir = cres.metadata['preserved_ir']
        return (func_ir, cres.fndesc)
    ignore_loops_ir, ignore_loops_fndesc = get_info(self.LoopIgnoringCompiler)
    canonicalise_loops_ir, canonicalise_loops_fndesc = get_info(self.LoopCanonicalisingCompiler)

    def compare_cfg(a, b):
        a_cfg = compute_cfg_from_blocks(flatten_labels(a.blocks))
        b_cfg = compute_cfg_from_blocks(flatten_labels(b.blocks))
        self.assertEqual(a_cfg, b_cfg)
    compare_cfg(ignore_loops_ir, canonicalise_loops_ir)
    self.assertEqual(len(ignore_loops_fndesc.calltypes) + 5, len(canonicalise_loops_fndesc.calltypes))