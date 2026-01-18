import typing
from selenium.common.exceptions import InvalidArgumentException
from selenium.webdriver.remote.webelement import WebElement
from .input_device import InputDevice
from .interaction import POINTER
from .interaction import POINTER_KINDS
def create_pointer_move(self, duration=DEFAULT_MOVE_DURATION, x: float=0, y: float=0, origin: typing.Optional[WebElement]=None, **kwargs):
    action = {'type': 'pointerMove', 'duration': duration, 'x': x, 'y': y, **kwargs}
    if isinstance(origin, WebElement):
        action['origin'] = {'element-6066-11e4-a52e-4f735466cecf': origin.id}
    elif origin is not None:
        action['origin'] = origin
    self.add_action(self._convert_keys(action))