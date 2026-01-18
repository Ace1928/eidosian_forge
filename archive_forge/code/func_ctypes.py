import ctypes
from functools import cached_property
from numba.core import compiler, registry
from numba.core.caching import NullCache, FunctionCache
from numba.core.dispatcher import _FunctionCompiler
from numba.core.typing import signature
from numba.core.typing.ctypes_utils import to_ctypes
from numba.core.compiler_lock import global_compiler_lock
@cached_property
def ctypes(self):
    """
        A ctypes function object representing the C callback.
        """
    ctypes_args = [to_ctypes(ty) for ty in self._sig.args]
    ctypes_restype = to_ctypes(self._sig.return_type)
    functype = ctypes.CFUNCTYPE(ctypes_restype, *ctypes_args)
    return functype(self.address)