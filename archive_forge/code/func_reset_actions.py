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
def reset_actions(self) -> None:
    """Clears actions that are already stored locally and on the remote
        end."""
    self.w3c_actions.clear_actions()
    for device in self.w3c_actions.devices:
        device.clear_actions()