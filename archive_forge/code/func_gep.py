import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
def gep(builder, ptr, *inds, **kws):
    """
    Emit a getelementptr instruction for the given pointer and indices.
    The indices can be LLVM values or Python int constants.
    """
    name = kws.pop('name', '')
    inbounds = kws.pop('inbounds', False)
    assert not kws
    idx = []
    for i in inds:
        if isinstance(i, int):
            ind = int32_t(i)
        else:
            ind = i
        idx.append(ind)
    return builder.gep(ptr, idx, name=name, inbounds=inbounds)