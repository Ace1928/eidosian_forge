from collections import namedtuple
from weakref import finalize as _finalize
from numba.core.runtime import nrtdynmod
from llvmlite import binding as ll
from numba.core.compiler_lock import global_compiler_lock
from numba.core.typing.typeof import typeof_impl
from numba.core import types, config
from numba.core.runtime import _nrt_python as _nrt
def get_allocation_stats(self):
    """
        Returns a namedtuple of (alloc, free, mi_alloc, mi_free) for count of
        each memory operations.
        """
    return _nrt_mstats(alloc=_nrt.memsys_get_stats_alloc(), free=_nrt.memsys_get_stats_free(), mi_alloc=_nrt.memsys_get_stats_mi_alloc(), mi_free=_nrt.memsys_get_stats_mi_free())