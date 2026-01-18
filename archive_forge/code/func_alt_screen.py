import sys
import time
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Union
from .segment import ControlCode, ControlType, Segment
@classmethod
def alt_screen(cls, enable: bool) -> 'Control':
    """Enable or disable alt screen."""
    if enable:
        return cls(ControlType.ENABLE_ALT_SCREEN, ControlType.HOME)
    else:
        return cls(ControlType.DISABLE_ALT_SCREEN)