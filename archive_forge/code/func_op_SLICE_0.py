import builtins
import collections
import dis
import operator
import logging
import textwrap
from numba.core import errors, ir, config
from numba.core.errors import NotDefinedError, UnsupportedError, error_extras
from numba.core.ir_utils import get_definition, guard
from numba.core.utils import (PYVERSION, BINOPS_TO_OPERATORS,
from numba.core.byteflow import Flow, AdaptDFA, AdaptCFA, BlockKind
from numba.core.unsafe import eh
from numba.cpython.unsafe.tuple import unpack_single_tuple
def op_SLICE_0(self, inst, base, res, slicevar, indexvar, nonevar):
    base = self.get(base)
    slicegv = ir.Global('slice', slice, loc=self.loc)
    self.store(value=slicegv, name=slicevar)
    nonegv = ir.Const(None, loc=self.loc)
    self.store(value=nonegv, name=nonevar)
    none = self.get(nonevar)
    index = ir.Expr.call(self.get(slicevar), (none, none), (), loc=self.loc)
    self.store(value=index, name=indexvar)
    expr = ir.Expr.getitem(base, self.get(indexvar), loc=self.loc)
    self.store(value=expr, name=res)