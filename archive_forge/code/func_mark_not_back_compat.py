from typing import Any, Dict
import textwrap
def mark_not_back_compat(fn):
    docstring = textwrap.dedent(getattr(fn, '__doc__', None) or '')
    docstring += '\n.. warning::\n    This API is experimental and is *NOT* backward-compatible.\n'
    fn.__doc__ = docstring
    _MARKED_WITH_COMPATIBILITY.setdefault(fn)
    return fn