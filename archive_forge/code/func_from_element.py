from typing import Union
from selenium.webdriver.remote.webelement import WebElement
from . import interaction
from .input_device import InputDevice
@classmethod
def from_element(cls, element: WebElement, x_offset: int=0, y_offset: int=0):
    return cls(element, x_offset, y_offset)