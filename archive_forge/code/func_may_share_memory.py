from cupy._core import _kernel
from cupy._core import _memory_range
from cupy._manipulation import join
from cupy._sorting import search
def may_share_memory(a, b, max_work=None):
    if max_work is None:
        return _memory_range.may_share_bounds(a, b)
    raise NotImplementedError('Only supported for `max_work` is `None`')