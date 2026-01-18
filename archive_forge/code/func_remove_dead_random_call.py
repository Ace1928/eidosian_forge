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
def remove_dead_random_call(rhs, lives, call_list):
    if len(call_list) == 3 and call_list[1:] == ['random', numpy]:
        return call_list[0] not in {'seed', 'shuffle'}
    return False