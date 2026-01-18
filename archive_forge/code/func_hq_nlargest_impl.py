import heapq as hq
from numba.core import types
from numba.core.errors import TypingError
from numba.core.extending import overload, register_jitable
def hq_nlargest_impl(n, iterable):
    if n == 0:
        return [iterable[0] for _ in range(0)]
    elif n == 1:
        out = max(iterable)
        return [out]
    size = len(iterable)
    if n >= size:
        return sorted(iterable)[::-1][:n]
    it = iter(iterable)
    result = [(elem, i) for i, elem in zip(range(0, -n, -1), it)]
    hq.heapify(result)
    top = result[0][0]
    order = -n
    for elem in it:
        if top < elem:
            hq.heapreplace(result, (elem, order))
            top, _order = result[0]
            order -= 1
    result.sort(reverse=True)
    return [elem for elem, order in result]