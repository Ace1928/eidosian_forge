from abc import ABCMeta, abstractmethod
from enum import Enum, auto
from typing import Dict, Optional
from pyglet import event
@abstractmethod
def get_default_input(self):
    """Returns a default active input device or None if none available."""
    pass