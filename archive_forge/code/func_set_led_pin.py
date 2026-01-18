from __future__ import annotations
import abc
import time
import typing
from .common import BaseScreen
def set_led_pin(self, led: Literal[0, 1, 2, 3], rg: Literal[0, 1], value: int) -> None:
    """
        led -- 0 to 3
        rg -- 0 for red, 1 for green
        value -- 0 to 100
        """
    if not 0 <= led <= 3:
        raise ValueError(led)
    if rg not in {0, 1}:
        raise ValueError(rg)
    if not 0 <= value <= 100:
        raise ValueError(value)
    self.queue_command(self.CMD_GPO, bytearray([12 - 2 * led - rg, value]))