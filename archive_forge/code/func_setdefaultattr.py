from __future__ import annotations
import itertools
import typing
import warnings
import weakref
def setdefaultattr(obj, name, value):
    if hasattr(obj, name):
        return getattr(obj, name)
    setattr(obj, name, value)
    return value