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
def maybe_list(obj, scalars=(Mapping, str)):
    """Return list of one element if ``l`` is a scalar."""
    return obj if obj is None or is_list(obj, scalars) else [obj]