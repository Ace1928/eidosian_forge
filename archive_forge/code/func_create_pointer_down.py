import typing
from selenium.common.exceptions import InvalidArgumentException
from selenium.webdriver.remote.webelement import WebElement
from .input_device import InputDevice
from .interaction import POINTER
from .interaction import POINTER_KINDS
def create_pointer_down(self, **kwargs):
    data = {'type': 'pointerDown', 'duration': 0, **kwargs}
    self.add_action(self._convert_keys(data))