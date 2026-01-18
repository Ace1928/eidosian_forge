from typing import Optional
from selenium.webdriver.remote.webelement import WebElement
from . import interaction
from .interaction import Interaction
from .mouse_button import MouseButton
from .pointer_input import PointerInput
def context_click(self, element: Optional[WebElement]=None):
    return self.click(element=element, button=MouseButton.RIGHT)