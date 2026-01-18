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
def on_click(self, handler: EventCallback) -> None:
    """ Set up a handler for button or menu item clicks.

        Args:
            handler (func) : handler function to call when button is activated.

        Returns:
            None

        """
    self.on_event(ButtonClick, handler)
    self.on_event(MenuItemClick, handler)