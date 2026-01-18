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
def is_namedtuple_class(c):
    """check if c is a namedtuple class"""
    if not isinstance(c, type):
        return False
    bases = c.__bases__
    if len(bases) != 1 or bases[0] != tuple:
        return False
    if not hasattr(c, '_make'):
        return False
    fields = getattr(c, '_fields', None)
    if not isinstance(fields, tuple):
        return False
    return all((isinstance(f, str) for f in fields))