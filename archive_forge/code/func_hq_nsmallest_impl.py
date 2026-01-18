import heapq as hq
from numba.core import types
from numba.core.errors import TypingError
from numba.core.extending import overload, register_jitable
def hq_nsmallest_impl(n, iterable):
    if n == 0:
        return [iterable[0] for _ in range(0)]
    elif n == 1:
        out = min(iterable)
        return [out]
    size = len(iterable)
    if n >= size:
        return sorted(iterable)[:n]
    it = iter(iterable)
    result = [(elem, i) for i, elem in zip(range(n), it)]
    _heapify_max(result)
    top = result[0][0]
    order = n
    for elem in it:
        if elem < top:
            _heapreplace_max(result, (elem, order))
            top, _order = result[0]
            order += 1
    result.sort()
    return [elem for elem, order in result]