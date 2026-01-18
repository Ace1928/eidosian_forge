from collections import defaultdict
from collections.abc import Sequence
import types as pytypes
import weakref
import threading
import contextlib
import operator
import numba
from numba.core import types, errors
from numba.core.typeconv import Conversion, rules
from numba.core.typing import templates
from numba.core.utils import order_by_target_specificity
from .typeof import typeof, Purpose
from numba.core import utils
def resolve_function_type(self, func, args, kws):
    """
        Resolve function type *func* for argument types *args* and *kws*.
        A signature is returned.
        """
    try:
        res = self._resolve_user_function_type(func, args, kws)
    except errors.TypingError as e:
        last_exception = e
        res = None
    else:
        last_exception = None
    if res is not None:
        return res
    res = self._resolve_builtin_function_type(func, args, kws)
    if res is None and last_exception is not None:
        raise last_exception
    return res