from typing import Sequence
from IPython.utils.docs import GENERATING_DOCUMENTATION
def flag_calls(func):
    """Wrap a function to detect and flag when it gets called.

    This is a decorator which takes a function and wraps it in a function with
    a 'called' attribute. wrapper.called is initialized to False.

    The wrapper.called attribute is set to False right before each call to the
    wrapped function, so if the call fails it remains False.  After the call
    completes, wrapper.called is set to True and the output is returned.

    Testing for truth in wrapper.called allows you to determine if a call to
    func() was attempted and succeeded."""
    if hasattr(func, 'called'):
        return func

    def wrapper(*args, **kw):
        wrapper.called = False
        out = func(*args, **kw)
        wrapper.called = True
        return out
    wrapper.called = False
    wrapper.__doc__ = func.__doc__
    return wrapper