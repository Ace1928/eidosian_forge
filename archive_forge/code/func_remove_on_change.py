from __future__ import annotations
import logging # isort:skip
from collections import defaultdict
from inspect import signature
from typing import (
from ..events import Event, ModelEvent
from ..util.functions import get_param_info
def remove_on_change(self, attr: str, *callbacks: PropertyCallback) -> None:
    """ Remove a callback from this object """
    if len(callbacks) == 0:
        raise ValueError('remove_on_change takes an attribute name and one or more callbacks, got only one parameter')
    _callbacks = self._callbacks.setdefault(attr, [])
    for callback in callbacks:
        _callbacks.remove(callback)