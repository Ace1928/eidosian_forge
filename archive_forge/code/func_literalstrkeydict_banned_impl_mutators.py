import ctypes
import operator
from enum import IntEnum
from llvmlite import ir
from numba import _helperlib
from numba.core.extending import (
from numba.core.imputils import iternext_impl, impl_ret_untracked
from numba.core import types, cgutils
from numba.core.types import (
from numba.core.imputils import impl_ret_borrowed, RefType
from numba.core.errors import TypingError, LoweringError
from numba.core import typing
from numba.typed.typedobjectutils import (_as_bytes, _cast, _nonoptional,
@overload_method(types.LiteralStrKeyDict, 'popitem')
@overload_method(types.LiteralStrKeyDict, 'pop')
@overload_method(types.LiteralStrKeyDict, 'clear')
@overload_method(types.LiteralStrKeyDict, 'setdefault')
@overload_method(types.LiteralStrKeyDict, 'update')
def literalstrkeydict_banned_impl_mutators(d, *args):
    if not isinstance(d, types.LiteralStrKeyDict):
        return
    raise TypingError('Cannot mutate a literal dictionary')