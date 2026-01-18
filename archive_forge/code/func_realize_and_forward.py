import functools
from typing import Optional
from .base import VariableTracker
@functools.wraps(getattr(VariableTracker, name))
def realize_and_forward(self, *args, **kwargs):
    return getattr(self.realize(), name)(*args, **kwargs)