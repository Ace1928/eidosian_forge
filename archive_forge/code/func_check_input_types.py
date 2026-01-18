import heapq as hq
from numba.core import types
from numba.core.errors import TypingError
from numba.core.extending import overload, register_jitable
def check_input_types(n, iterable):
    if not isinstance(n, (types.Integer, types.Boolean)):
        raise TypingError("First argument 'n' must be an integer")
    if not isinstance(iterable, (types.Sequence, types.Array, types.ListType)):
        raise TypingError("Second argument 'iterable' must be iterable")