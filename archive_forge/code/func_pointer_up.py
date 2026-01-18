from typing import Optional
from selenium.webdriver.remote.webelement import WebElement
from . import interaction
from .interaction import Interaction
from .mouse_button import MouseButton
from .pointer_input import PointerInput
def pointer_up(self, button=MouseButton.LEFT):
    self._button_action('create_pointer_up', button=button)
    return self