from typing import Any, Dict
import textwrap
def mark_back_compat(fn):
    docstring = textwrap.dedent(getattr(fn, '__doc__', None) or '')
    docstring += '\n.. note::\n    Backwards-compatibility for this API is guaranteed.\n'
    fn.__doc__ = docstring
    _BACK_COMPAT_OBJECTS.setdefault(fn)
    _MARKED_WITH_COMPATIBILITY.setdefault(fn)
    return fn