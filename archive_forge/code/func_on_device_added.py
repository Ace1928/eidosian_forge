from abc import ABCMeta, abstractmethod
from enum import Enum, auto
from typing import Dict, Optional
from pyglet import event
def on_device_added(self, device: AudioDevice):
    """Event, occurs when a new device is added to the system."""
    pass