from __future__ import annotations
import typing as t
from weakref import ref
from blinker._saferef import BoundMethodWeakref
def hashable_identity(obj: object) -> IdentityType:
    if hasattr(obj, '__func__'):
        return (id(obj.__func__), id(obj.__self__))
    elif hasattr(obj, 'im_func'):
        return (id(obj.im_func), id(obj.im_self))
    elif isinstance(obj, (int, str)):
        return obj
    else:
        return id(obj)