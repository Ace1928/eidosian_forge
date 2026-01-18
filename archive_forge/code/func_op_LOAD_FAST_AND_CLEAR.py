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
def op_LOAD_FAST_AND_CLEAR(self, inst, res):
    try:
        srcname = self.code_locals[inst.arg]
        self.store(value=self.get(srcname), name=res)
    except NotDefinedError:
        undef = ir.Expr.undef(loc=self.loc)
        self.store(undef, name=res)