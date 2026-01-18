import heapq as hq
from numba.core import types
from numba.core.errors import TypingError
from numba.core.extending import overload, register_jitable
@register_jitable
def reversed_range(x):
    return range(x - 1, -1, -1)