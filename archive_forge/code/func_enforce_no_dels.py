import numpy
import math
import types as pytypes
import collections
import warnings
import numba
from numba.core.extending import _Intrinsic
from numba.core import types, typing, ir, analysis, postproc, rewrites, config
from numba.core.typing.templates import signature
from numba.core.analysis import (compute_live_map, compute_use_defs,
from numba.core.errors import (TypingError, UnsupportedError,
import copy
def enforce_no_dels(func_ir):
    """
    Enforce there being no ir.Del nodes in the IR.
    """
    for blk in func_ir.blocks.values():
        dels = [x for x in blk.find_insts(ir.Del)]
        if dels:
            msg = 'Illegal IR, del found at: %s' % dels[0]
            raise CompilerError(msg, loc=dels[0].loc)