from __future__ import annotations
import copy
import math
import operator
import typing as t
from contextvars import ContextVar
from functools import partial
from functools import update_wrapper
from operator import attrgetter
from .wsgi import ClosingIterator
def bind_f(instance: LocalProxy, obj: t.Any) -> t.Callable:
    return partial(f, obj)