import inspect
import sys
def fix_call(callable, *args, **kw):
    """
    Call ``callable(*args, **kw)`` fixing any type errors that come out.
    """
    try:
        val = callable(*args, **kw)
    except TypeError:
        exc_info = fix_type_error(None, callable, args, kw)
        raise exc_info[1] from None
    return val