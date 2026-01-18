from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Callable
from ...core.enums import ButtonType
from ...core.has_props import HasProps, abstract
from ...core.properties import (
from ...events import ButtonClick, MenuItemClick
from ..callbacks import Callback
from ..dom import DOMNode
from ..ui.icons import BuiltinIcon, Icon
from ..ui.tooltips import Tooltip
from .widget import Widget
def js_on_click(self, handler: Callback) -> None:
    """ Set up a JavaScript handler for button or menu item clicks. """
    self.js_on_event(ButtonClick, handler)
    self.js_on_event(MenuItemClick, handler)