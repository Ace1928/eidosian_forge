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
def move_to_element_with_offset(self, to_element: WebElement, xoffset: int, yoffset: int) -> ActionChains:
    """Move the mouse by an offset of the specified element. Offsets are
        relative to the in-view center point of the element.

        :Args:
         - to_element: The WebElement to move to.
         - xoffset: X offset to move to, as a positive or negative integer.
         - yoffset: Y offset to move to, as a positive or negative integer.
        """
    self.w3c_actions.pointer_action.move_to(to_element, int(xoffset), int(yoffset))
    self.w3c_actions.key_action.pause()
    return self