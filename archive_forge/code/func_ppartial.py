from .abstract import Thenable
from .promises import promise
def ppartial(p, *args, **kwargs):
    """Create/modify promise with partial arguments."""
    p = ensure_promise(p)
    if args:
        p.args = args + p.args
    if kwargs:
        p.kwargs.update(kwargs)
    return p