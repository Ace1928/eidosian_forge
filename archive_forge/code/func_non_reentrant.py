import threading
import scipy._lib.decorator
def non_reentrant(err_msg=None):
    """
    Decorate a function with a threading lock and prevent reentrant calls.
    """

    def decorator(func):
        msg = err_msg
        if msg is None:
            msg = '%s is not re-entrant' % func.__name__
        lock = ReentrancyLock(msg)
        return lock.decorate(func)
    return decorator