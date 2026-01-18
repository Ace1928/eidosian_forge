import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
def gep_inbounds(builder, ptr, *inds, **kws):
    """
    Same as *gep*, but add the `inbounds` keyword.
    """
    return gep(builder, ptr, *inds, inbounds=True, **kws)