import heapq as hq
from numba.core import types
from numba.core.errors import TypingError
from numba.core.extending import overload, register_jitable
def assert_item_type_consistent_with_heap_type(heap, item):
    if not heap.dtype == item:
        raise TypingError('heap type must be the same as item type')