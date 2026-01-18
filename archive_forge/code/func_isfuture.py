import reprlib
from _thread import get_ident
from . import format_helpers
def isfuture(obj):
    """Check for a Future.

    This returns True when obj is a Future instance or is advertising
    itself as duck-type compatible by setting _asyncio_future_blocking.
    See comment in Future for more details.
    """
    return hasattr(obj.__class__, '_asyncio_future_blocking') and obj._asyncio_future_blocking is not None