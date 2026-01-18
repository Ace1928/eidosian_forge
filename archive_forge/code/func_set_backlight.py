from __future__ import annotations
import abc
import time
import typing
from .common import BaseScreen
def set_backlight(self, value: int) -> None:
    """
        Set backlight brightness

        value -- 0 to 100
        """
    if not 0 <= value <= 100:
        raise ValueError(value)
    self.queue_command(self.CMD_BACKLIGHT, bytearray([value]))