import heapq as hq
from numba.core import types
from numba.core.errors import TypingError
from numba.core.extending import overload, register_jitable
@overload(hq.heappop)
def hq_heappop(heap):
    assert_heap_type(heap)

    def hq_heappop_impl(heap):
        lastelt = heap.pop()
        if heap:
            returnitem = heap[0]
            heap[0] = lastelt
            _siftup(heap, 0)
            return returnitem
        return lastelt
    return hq_heappop_impl