import functools
from collections import namedtuple
from llvmlite import ir
from numba.core import types, cgutils, errors, config
from numba.core.utils import PYVERSION
def eh_try(self, builder):
    """Begin a try-block.
        """
    ctx = self._context
    cc = ctx.call_conv
    cc.set_try_status(builder)