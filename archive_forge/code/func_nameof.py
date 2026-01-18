import inspect
import os
import sys
def nameof(obj, quote=False):
    """Returns the most descriptive name of a Python module, class, or function,
    as a Unicode string

    If quote=True, name is quoted with repr().

    Best-effort, but guaranteed to not fail - always returns something.
    """
    try:
        name = obj.__qualname__
    except Exception:
        try:
            name = obj.__name__
        except Exception:
            try:
                name = repr(obj)
            except Exception:
                return '<unknown>'
            else:
                quote = False
    if quote:
        try:
            name = repr(name)
        except Exception:
            pass
    return force_str(name, 'utf-8', 'replace')