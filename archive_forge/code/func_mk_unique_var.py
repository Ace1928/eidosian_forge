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
def mk_unique_var(prefix):
    global _unique_var_count
    var = prefix + '.' + str(_unique_var_count)
    _unique_var_count = _unique_var_count + 1
    return var