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
def test_literal_unroll_globals_and_locals(self):
    f = 'def foo():\n\tfor x in literal_unroll((1,)):\n\t\tpass'
    l = {}
    exec(f, {}, l)
    foo = njit(pipeline_class=CapturingCompiler)(l['foo'])
    with self.assertRaises(errors.TypingError) as raises:
        foo()
    self.assertIn("Untyped global name 'literal_unroll'", str(raises.exception))
    l = {}
    exec(f, {'literal_unroll': literal_unroll}, l)
    foo = njit(pipeline_class=CapturingCompiler)(l['foo'])
    foo()
    cres = foo.overloads[foo.signatures[0]]
    self.assertTrue(cres.metadata['mutation_results'][LiteralUnroll])
    from textwrap import dedent
    f = '\n            def gen():\n                from numba import literal_unroll\n                def foo():\n                    for x in literal_unroll((1,)):\n                        pass\n                return foo\n            bar = gen()\n            '
    l = {}
    exec(dedent(f), {}, l)
    foo = njit(pipeline_class=CapturingCompiler)(l['bar'])
    foo()
    cres = foo.overloads[foo.signatures[0]]
    self.assertTrue(cres.metadata['mutation_results'][LiteralUnroll])
    from textwrap import dedent
    f = '\n            def gen():\n                from numba import literal_unroll as something_else\n                def foo():\n                    for x in something_else((1,)):\n                        pass\n                return foo\n            bar = gen()\n            '
    l = {}
    exec(dedent(f), {}, l)
    foo = njit(pipeline_class=CapturingCompiler)(l['bar'])
    foo()
    cres = foo.overloads[foo.signatures[0]]
    self.assertTrue(cres.metadata['mutation_results'][LiteralUnroll])