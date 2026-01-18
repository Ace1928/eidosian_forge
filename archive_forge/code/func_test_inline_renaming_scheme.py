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
def test_inline_renaming_scheme(self):

    @njit(inline='always')
    def bar(z):
        x = 5
        y = 10
        return x + y + z

    @njit(pipeline_class=IRPreservingTestPipeline)
    def foo(a, b):
        return (bar(a), bar(b))
    self.assertEqual(foo(10, 20), (25, 35))
    func_ir = foo.overloads[foo.signatures[0]].metadata['preserved_ir']
    store = []
    for blk in func_ir.blocks.values():
        for stmt in blk.body:
            if isinstance(stmt, ir.Assign):
                if isinstance(stmt.value, ir.Const):
                    if stmt.value.value == 5:
                        store.append(stmt)
    self.assertEqual(len(store), 2)
    for i in store:
        name = i.target.name
        basename = self.id().lstrip(self.__module__)
        regex = f'{basename}__locals__bar_v[0-9]+.x'
        self.assertRegex(name, regex)