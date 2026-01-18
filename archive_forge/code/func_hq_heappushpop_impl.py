import heapq as hq
from numba.core import types
from numba.core.errors import TypingError
from numba.core.extending import overload, register_jitable
def hq_heappushpop_impl(heap, item):
    if heap and heap[0] < item:
        item, heap[0] = (heap[0], item)
        _siftup(heap, 0)
    return item