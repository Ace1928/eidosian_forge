import collections
import functools
import sys
import types as pytypes
import uuid
import weakref
from contextlib import ExitStack
from abc import abstractmethod
from numba import _dispatcher
from numba.core import (
from numba.core.compiler_lock import global_compiler_lock
from numba.core.typeconv.rules import default_type_manager
from numba.core.typing.templates import fold_arguments
from numba.core.typing.typeof import Purpose, typeof
from numba.core.bytecode import get_code_object
from numba.core.caching import NullCache, FunctionCache
from numba.core import entrypoints
from numba.core.retarget import BaseRetarget
import numba.core.event as ev
def get_compile_result(self, sig):
    """Compile (if needed) and return the compilation result with the
        given signature.

        Returns ``CompileResult``.
        Raises ``NumbaError`` if the signature is incompatible.
        """
    atypes = tuple(sig.args)
    if atypes not in self.overloads:
        if self._can_compile:
            self.compile(atypes)
        else:
            msg = f'{sig} not available and compilation disabled'
            raise errors.TypingError(msg)
    return self.overloads[atypes]