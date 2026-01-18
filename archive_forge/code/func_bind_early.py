from __future__ import annotations
from heapq import heappop, heappush
from itertools import count
from types import MethodType
from typing import (
from simpy.events import (
@staticmethod
def bind_early(instance: object) -> None:
    """Bind all :class:`BoundClass` attributes of the *instance's* class
        to the instance itself to increase performance."""
    for name, obj in instance.__class__.__dict__.items():
        if type(obj) is BoundClass:
            bound_class = getattr(instance, name)
            setattr(instance, name, bound_class)