from __future__ import annotations
import logging # isort:skip
import copy
from typing import (
import numpy as np
def notify_owner(func):
    """ A decorator for mutating methods of property container classes
    that notifies owners of the property container about mutating changes.

    Args:
        func (callable) : the container method to wrap in a notification

    Returns:
        wrapped method

    Examples:

        A ``__setitem__`` could be wrapped like this:

        .. code-block:: python

            # x[i] = y
            @notify_owner
            def __setitem__(self, i, y):
                return super().__setitem__(i, y)

    The returned wrapped method will have a docstring indicating what
    original method it is wrapping.

    """

    def wrapper(self, *args, **kwargs):
        old = self._saved_copy()
        result = func(self, *args, **kwargs)
        self._notify_owners(old)
        return result
    wrapper.__doc__ = f'Container method ``{func.__name__}`` instrumented to notify property owners'
    return wrapper