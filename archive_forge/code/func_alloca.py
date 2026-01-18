import builtins
import operator
import inspect
from functools import cached_property
import llvmlite.ir
from numba.core import types, utils, ir, generators, cgutils
from numba.core.errors import (ForbiddenConstruct, LoweringError,
from numba.core.lowering import BaseLower
def alloca(self, name, ltype=None):
    """
        Allocate a stack slot and initialize it to NULL.
        The default is to allocate a pyobject pointer.
        Use ``ltype`` to override.
        """
    if ltype is None:
        ltype = self.context.get_value_type(types.pyobject)
    with self.builder.goto_block(self.entry_block):
        ptr = self.builder.alloca(ltype, name=name)
        self.builder.store(cgutils.get_null_value(ltype), ptr)
    return ptr