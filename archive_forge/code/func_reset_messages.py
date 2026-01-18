import random
import threading
import time
from .messages import Message
from .parser import Parser
def reset_messages():
    """Yield "All Notes Off" and "Reset All Controllers" for all channels"""
    ALL_NOTES_OFF = 123
    RESET_ALL_CONTROLLERS = 121
    for channel in range(16):
        for control in [ALL_NOTES_OFF, RESET_ALL_CONTROLLERS]:
            yield Message('control_change', channel=channel, control=control)