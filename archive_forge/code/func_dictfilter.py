from __future__ import annotations
import inspect
import random
import threading
from collections import OrderedDict, UserDict
from collections.abc import Iterable, Mapping
from itertools import count, repeat
from time import sleep, time
from vine.utils import wraps
from .encoding import safe_repr as _safe_repr
def dictfilter(d=None, **kw):
    """Remove all keys from dict ``d`` whose value is :const:`None`."""
    d = kw if d is None else dict(d, **kw) if kw else d
    return {k: v for k, v in d.items() if v is not None}