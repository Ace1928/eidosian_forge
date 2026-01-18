from .abstract import Thenable
from .promises import promise
def ensure_promise(p):
    """Ensure p is a promise.

    If p is not a promise, a new promise is created with p' as callback.
    """
    if p is None:
        return promise()
    return maybe_promise(p)