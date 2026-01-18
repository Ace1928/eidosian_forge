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
def fxrangemax(start=1.0, stop=None, step=1.0, max=100.0):
    sum_, cur = (0, start * 1.0)
    while 1:
        if sum_ >= max:
            break
        yield cur
        if stop:
            cur = min(cur + step, stop)
        else:
            cur += step
        sum_ += cur