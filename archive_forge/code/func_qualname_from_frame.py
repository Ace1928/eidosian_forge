from __future__ import annotations
from types import FrameType
from typing import cast, Callable, Sequence
def qualname_from_frame(frame: FrameType) -> str | None:
    """Get a qualified name for the code running in `frame`."""
    co = frame.f_code
    fname = co.co_name
    method = None
    if co.co_argcount and co.co_varnames[0] == 'self':
        self = frame.f_locals.get('self', None)
        method = getattr(self, fname, None)
    if method is None:
        func = frame.f_globals.get(fname)
        if func is None:
            return None
        return cast(str, func.__module__ + '.' + fname)
    func = getattr(method, '__func__', None)
    if func is None:
        cls = self.__class__
        return cast(str, cls.__module__ + '.' + cls.__name__ + '.' + fname)
    return cast(str, func.__module__ + '.' + func.__qualname__)