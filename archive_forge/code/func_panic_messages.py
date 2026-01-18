import random
import threading
import time
from .messages import Message
from .parser import Parser
def panic_messages():
    """Yield "All Sounds Off" for all channels.

    This will mute all sounding notes regardless of
    envelopes. Useful when notes are hanging and nothing else
    helps.
    """
    ALL_SOUNDS_OFF = 120
    for channel in range(16):
        yield Message('control_change', channel=channel, control=ALL_SOUNDS_OFF)