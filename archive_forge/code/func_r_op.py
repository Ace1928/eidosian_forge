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
def r_op(obj: t.Any, other: t.Any) -> t.Any:
    return op(other, obj)