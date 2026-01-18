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
def parallel_diagnostics(self, signature=None, level=1):
    """
        Print parallel diagnostic information for the given signature. If no
        signature is present it is printed for all known signatures. level is
        used to adjust the verbosity, level=1 (default) is minimal verbosity,
        and 2, 3, and 4 provide increasing levels of verbosity.
        """

    def dump(sig):
        ol = self.overloads[sig]
        pfdiag = ol.metadata.get('parfor_diagnostics', None)
        if pfdiag is None:
            msg = "No parfors diagnostic available, is 'parallel=True' set?"
            raise ValueError(msg)
        pfdiag.dump(level)
    if signature is not None:
        dump(signature)
    else:
        [dump(sig) for sig in self.signatures]