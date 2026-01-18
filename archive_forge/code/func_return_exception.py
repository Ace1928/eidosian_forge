from collections import namedtuple, defaultdict
import operator
import warnings
from functools import partial
import llvmlite.ir
from llvmlite.ir import Constant, IRBuilder
from numba.core import (typing, utils, types, ir, debuginfo, funcdesc,
from numba.core.errors import (LoweringError, new_error_context, TypingError,
from numba.core.funcdesc import default_mangler
from numba.core.environment import Environment
from numba.core.analysis import compute_use_defs, must_use_alloca
from numba.misc.firstlinefinder import get_func_body_first_lineno
def return_exception(self, exc_class, exc_args=None, loc=None):
    """Propagate exception to the caller.
        """
    self.call_conv.return_user_exc(self.builder, exc_class, exc_args, loc=loc, func_name=self.func_ir.func_id.func_name)