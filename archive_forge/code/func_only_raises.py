from . import trace
def only_raises(*errors):
    """Make a decorator that will only allow the given error classes to be
    raised.  All other errors will be logged and then discarded.

    Typical use is something like::

        @only_raises(LockNotHeld, LockBroken)
        def unlock(self):
            # etc
    """

    def decorator(unbound):

        def wrapped(*args, **kwargs):
            try:
                return unbound(*args, **kwargs)
            except errors:
                raise
            except:
                trace.mutter('Error suppressed by only_raises:')
                trace.log_exception_quietly()
        wrapped.__doc__ = unbound.__doc__
        wrapped.__name__ = unbound.__name__
        return wrapped
    return decorator