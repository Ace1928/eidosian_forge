import functools
from collections import namedtuple
from llvmlite import ir
from numba.core import types, cgutils, errors, config
from numba.core.utils import PYVERSION
def eh_check(self, builder):
    """Check if an exception is raised
        """
    ctx = self._context
    cc = ctx.call_conv
    trystatus = cc.check_try_status(builder)
    excinfo = trystatus.excinfo
    has_raised = builder.not_(cgutils.is_null(builder, excinfo))
    if PYVERSION < (3, 11):
        with builder.if_then(has_raised):
            self.eh_end_try(builder)
    return has_raised