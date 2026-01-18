from __future__ import annotations
import typing as t
from weakref import ref
from blinker._saferef import BoundMethodWeakref
def callable_reference(object, callback=None):
    """Return an annotated weak ref, supporting bound instance methods."""
    if hasattr(object, 'im_self') and object.im_self is not None:
        return BoundMethodWeakref(target=object, on_delete=callback)
    elif hasattr(object, '__self__') and object.__self__ is not None:
        return BoundMethodWeakref(target=object, on_delete=callback)
    return annotatable_weakref(object, callback)