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
def drag_and_drop_by_offset(self, source: WebElement, xoffset: int, yoffset: int) -> ActionChains:
    """Holds down the left mouse button on the source element, then moves
        to the target offset and releases the mouse button.

        :Args:
         - source: The element to mouse down.
         - xoffset: X offset to move to.
         - yoffset: Y offset to move to.
        """
    self.click_and_hold(source)
    self.move_by_offset(xoffset, yoffset)
    self.release()
    return self