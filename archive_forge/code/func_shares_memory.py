from cupy._core import _kernel
from cupy._core import _memory_range
from cupy._manipulation import join
from cupy._sorting import search
def shares_memory(a, b, max_work=None):
    if a is b and a.size != 0:
        return True
    if max_work == 'MAY_SHARE_BOUNDS':
        return _memory_range.may_share_bounds(a, b)
    if max_work in (None, 'MAY_SHARE_EXACT'):
        a_ptrs = _get_memory_ptrs(a).ravel()
        b_ptrs = _get_memory_ptrs(b).reshape(-1, 1)
        a_ptrs.sort()
        x = search.searchsorted(a_ptrs, b_ptrs, 'left')
        y = search.searchsorted(a_ptrs, b_ptrs, 'right')
        return bool((x != y).any())
    raise NotImplementedError('Not supported for integer `max_work`.')