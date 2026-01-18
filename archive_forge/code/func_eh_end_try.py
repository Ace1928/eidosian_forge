import functools
from collections import namedtuple
from llvmlite import ir
from numba.core import types, cgutils, errors, config
from numba.core.utils import PYVERSION
def eh_end_try(self, builder):
    """End a try-block
        """
    ctx = self._context
    cc = ctx.call_conv
    cc.unset_try_status(builder)