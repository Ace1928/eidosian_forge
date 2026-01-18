import builtins
import operator
import inspect
from functools import cached_property
import llvmlite.ir
from numba.core import types, utils, ir, generators, cgutils
from numba.core.errors import (ForbiddenConstruct, LoweringError,
from numba.core.lowering import BaseLower
def check_int_status(self, num, ok_value=0):
    """
        Raise an exception if *num* is smaller than *ok_value*.
        """
    ok = llvmlite.ir.Constant(num.type, ok_value)
    pred = self.builder.icmp_signed('<', num, ok)
    with cgutils.if_unlikely(self.builder, pred):
        self.return_exception_raised()