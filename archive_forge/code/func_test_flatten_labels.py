import numba
from numba.tests.support import TestCase, unittest
from numba.core.registry import cpu_target
from numba.core.compiler import CompilerBase, Flags
from numba.core.compiler_machinery import PassManager
from numba.core import types, ir, bytecode, compiler, ir_utils, registry
from numba.core.untyped_passes import (ExtractByteCode, TranslateByteCode,
from numba.core.typed_passes import (NopythonTypeInference,
from numba.experimental import jitclass
def test_flatten_labels(self):
    """ tests flatten_labels """

    def foo(a):
        acc = 0
        if a > 3:
            acc += 1
            if a > 19:
                return 53
        elif a < 1000:
            if a >= 12:
                acc += 1
            for x in range(10):
                acc -= 1
                if acc < 2:
                    break
            else:
                acc += 7
        else:
            raise ValueError('some string')
        py310_defeat1 = 1
        py310_defeat2 = 2
        py310_defeat3 = 3
        py310_defeat4 = 4
        return acc

    def bar(a):
        acc = 0
        z = 12
        if a > 3:
            acc += 1
            z += 12
            if a > 19:
                z += 12
                return 53
        elif a < 1000:
            if a >= 12:
                z += 12
                acc += 1
            for x in range(10):
                z += 12
                acc -= 1
                if acc < 2:
                    break
            else:
                z += 12
                acc += 7
        else:
            raise ValueError('some string')
        py310_defeat1 = 1
        py310_defeat2 = 2
        py310_defeat3 = 3
        py310_defeat4 = 4
        return acc

    def baz(a):
        acc = 0
        if a > 3:
            acc += 1
            if a > 19:
                return 53
            else:
                return 55
        elif a < 1000:
            if a >= 12:
                acc += 1
            for x in range(10):
                acc -= 1
                if acc < 2:
                    break
            else:
                acc += 7
        else:
            raise ValueError('some string')
        py310_defeat1 = 1
        py310_defeat2 = 2
        py310_defeat3 = 3
        py310_defeat4 = 4
        return acc

    def get_flat_cfg(func):
        func_ir = ir_utils.compile_to_numba_ir(func, dict())
        flat_blocks = ir_utils.flatten_labels(func_ir.blocks)
        self.assertEqual(max(flat_blocks.keys()) + 1, len(func_ir.blocks))
        return ir_utils.compute_cfg_from_blocks(flat_blocks)
    foo_cfg = get_flat_cfg(foo)
    bar_cfg = get_flat_cfg(bar)
    baz_cfg = get_flat_cfg(baz)
    self.assertEqual(foo_cfg, bar_cfg)
    self.assertNotEqual(foo_cfg, baz_cfg)