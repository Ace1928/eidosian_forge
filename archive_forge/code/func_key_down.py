from __future__ import annotations
from ..utils import keys_to_typing
from .interaction import KEY
from .interaction import Interaction
from .key_input import KeyInput
from .pointer_input import PointerInput
from .wheel_input import WheelInput
def key_down(self, letter: str) -> KeyActions:
    return self._key_action('create_key_down', letter)