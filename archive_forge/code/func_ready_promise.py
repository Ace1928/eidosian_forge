from .abstract import Thenable
from .promises import promise
def ready_promise(callback=None, *args):
    """Create promise that is already fulfilled."""
    p = ensure_promise(callback)
    p(*args)
    return p