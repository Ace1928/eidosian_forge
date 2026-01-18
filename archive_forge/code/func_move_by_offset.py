from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Union
from selenium.webdriver.remote.webelement import WebElement
from .actions.action_builder import ActionBuilder
from .actions.key_input import KeyInput
from .actions.pointer_input import PointerInput
from .actions.wheel_input import ScrollOrigin
from .actions.wheel_input import WheelInput
from .utils import keys_to_typing
def move_by_offset(self, xoffset: int, yoffset: int) -> ActionChains:
    """Moving the mouse to an offset from current mouse position.

        :Args:
         - xoffset: X offset to move to, as a positive or negative integer.
         - yoffset: Y offset to move to, as a positive or negative integer.
        """
    self.w3c_actions.pointer_action.move_by(xoffset, yoffset)
    self.w3c_actions.key_action.pause()
    return self