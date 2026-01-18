from typing import Optional
from selenium.webdriver.remote.webelement import WebElement
from . import interaction
from .interaction import Interaction
from .mouse_button import MouseButton
from .pointer_input import PointerInput
def double_click(self, element: Optional[WebElement]=None):
    if element:
        self.move_to(element)
    self.pointer_down(MouseButton.LEFT)
    self.pointer_up(MouseButton.LEFT)
    self.pointer_down(MouseButton.LEFT)
    self.pointer_up(MouseButton.LEFT)
    return self