from numba.core.types.abstract import Callable, Literal, Type, Hashable
from numba.core.types.common import (Dummy, IterableType, Opaque,
from numba.core.typeconv import Conversion
from numba.core.errors import TypingError, LiteralTypingError
from numba.core.ir import UndefinedType
from numba.core.utils import get_hashable_key
def maybe_literal(value):
    """Get a Literal type for the value or None.
    """
    try:
        return literal(value)
    except LiteralTypingError:
        return