import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
def as_bool_bit(builder, value):
    return builder.icmp_unsigned('!=', value, value.type(0))