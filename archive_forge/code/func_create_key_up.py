from . import interaction
from .input_device import InputDevice
from .interaction import Interaction
from .interaction import Pause
def create_key_up(self, key) -> None:
    self.add_action(TypingInteraction(self, 'keyUp', key))