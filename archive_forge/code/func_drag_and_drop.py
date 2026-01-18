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
def drag_and_drop(self, source: WebElement, target: WebElement) -> ActionChains:
    """Holds down the left mouse button on the source element, then moves
        to the target element and releases the mouse button.

        :Args:
         - source: The element to mouse down.
         - target: The element to mouse up.
        """
    self.click_and_hold(source)
    self.release(target)
    return self