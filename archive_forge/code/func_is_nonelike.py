import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
def is_nonelike(ty):
    """ returns if 'ty' is none """
    return ty is None or isinstance(ty, types.NoneType) or isinstance(ty, types.Omitted)