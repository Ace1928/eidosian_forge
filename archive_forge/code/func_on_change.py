from __future__ import annotations
import logging # isort:skip
from collections import defaultdict
from inspect import signature
from typing import (
from ..events import Event, ModelEvent
from ..util.functions import get_param_info
def on_change(self, attr: str, *callbacks: PropertyCallback) -> None:
    """ Add a callback on this object to trigger when ``attr`` changes.

        Args:
            attr (str) : an attribute name on this object
            callback (callable) : a callback function to register

        Returns:
            None

        """
    if len(callbacks) == 0:
        raise ValueError('on_change takes an attribute name and one or more callbacks, got only one parameter')
    _callbacks = self._callbacks.setdefault(attr, [])
    for callback in callbacks:
        if callback in _callbacks:
            continue
        _check_callback(callback, ('attr', 'old', 'new'))
        _callbacks.append(callback)