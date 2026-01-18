from collections.abc import MutableSequence
from numba.core.types import ListType
from numba.core.imputils import numba_typeref_ctor
from numba.core.dispatcher import Dispatcher
from numba.core import types, config, cgutils
from numba import njit, typeof
from numba.core.extending import (
from numba.typed import listobject
from numba.core.errors import TypingError, LoweringError
from numba.core.typing.templates import Signature
import typing as pt
@classmethod
def empty_list(cls, item_type, allocated=DEFAULT_ALLOCATED):
    """Create a new empty List.

        Parameters
        ----------
        item_type: Numba type
            type of the list item.
        allocated: int
            number of items to pre-allocate
        """
    if config.DISABLE_JIT:
        return list()
    else:
        return cls(lsttype=ListType(item_type), allocated=allocated)