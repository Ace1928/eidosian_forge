import builtins
import operator
import inspect
from functools import cached_property
import llvmlite.ir
from numba.core import types, utils, ir, generators, cgutils
from numba.core.errors import (ForbiddenConstruct, LoweringError,
from numba.core.lowering import BaseLower
def check_occurred(self):
    """
        Return if an exception occurred.
        """
    err_occurred = cgutils.is_not_null(self.builder, self.pyapi.err_occurred())
    with cgutils.if_unlikely(self.builder, err_occurred):
        self.return_exception_raised()