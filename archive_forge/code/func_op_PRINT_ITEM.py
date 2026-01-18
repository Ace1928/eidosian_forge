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
def op_PRINT_ITEM(self, inst, item, printvar, res):
    item = self.get(item)
    printgv = ir.Global('print', print, loc=self.loc)
    self.store(value=printgv, name=printvar)
    call = ir.Expr.call(self.get(printvar), (item,), (), loc=self.loc)
    self.store(value=call, name=res)