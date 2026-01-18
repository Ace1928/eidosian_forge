from __future__ import annotations
from typing import (
import param
from bokeh.events import ButtonClick, MenuItemClick
from bokeh.models import Dropdown as _BkDropdown, Toggle as _BkToggle
from bokeh.models.ui import SVGIcon, TablerIcon
from ..io.resources import CDN_DIST
from ..links import Callback
from ..models.widgets import Button as _BkButton
from ._mixin import TooltipMixin
from .base import Widget
def jscallback(self, args: Dict[str, Any]={}, **callbacks: str) -> Callback:
    """
        Allows defining a Javascript (JS) callback to be triggered when a property
        changes on the source object. The keyword arguments define the
        properties that trigger a callback and the JS code that gets
        executed.

        Arguments
        ----------
        args: dict
          A mapping of objects to make available to the JS callback
        **callbacks: dict
          A mapping between properties on the source model and the code
          to execute when that property changes

        Returns
        -------
        callback: Callback
          The Callback which can be used to disable the callback.
        """
    for k, v in list(callbacks.items()):
        if k == 'clicks':
            k = 'event:' + self._event
        val = self._rename.get(v, v)
        if val is not None:
            callbacks[k] = val
    return Callback(self, code=callbacks, args=args)