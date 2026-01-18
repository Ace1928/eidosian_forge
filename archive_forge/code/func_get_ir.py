import collections
import weakref
import gc
import operator
from itertools import takewhile
import unittest
from numba import njit, jit
from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.untyped_passes import PreserveIR
from numba.core.typed_passes import IRLegalization
from numba.core import types, ir
from numba.tests.support import TestCase, override_config, SerialMixin
def get_ir(**options):

    class IRPreservingCompiler(CompilerBase):

        def define_pipelines(self):
            pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
            pm.add_pass_after(PreserveIR, IRLegalization)
            pm.finalize()
            return [pm]

    @njit(pipeline_class=IRPreservingCompiler, **options)
    def foo():
        a = 10
        b = 20
        c = a + b
        d = c / c
        return d
    foo()
    cres = foo.overloads[foo.signatures[0]]
    func_ir = cres.metadata['preserved_ir']
    return func_ir