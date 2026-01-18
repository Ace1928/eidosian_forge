import signal
import weakref
from functools import wraps
def removeHandler(method=None):
    if method is not None:

        @wraps(method)
        def inner(*args, **kwargs):
            initial = signal.getsignal(signal.SIGINT)
            removeHandler()
            try:
                return method(*args, **kwargs)
            finally:
                signal.signal(signal.SIGINT, initial)
        return inner
    global _interrupt_handler
    if _interrupt_handler is not None:
        signal.signal(signal.SIGINT, _interrupt_handler.original_handler)