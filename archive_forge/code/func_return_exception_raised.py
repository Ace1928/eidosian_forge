import builtins
import operator
import inspect
from functools import cached_property
import llvmlite.ir
from numba.core import types, utils, ir, generators, cgutils
from numba.core.errors import (ForbiddenConstruct, LoweringError,
from numba.core.lowering import BaseLower
def return_exception_raised(self):
    """
        Return with the currently raised exception.
        """
    self.cleanup_vars()
    self.call_conv.return_exc(self.builder)