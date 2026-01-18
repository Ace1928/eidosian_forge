from __future__ import annotations
import abc
import time
import typing
from .common import BaseScreen
def sent_event(self) -> None:
    """
        Cakk this method when you have sent a key repeat event so the
        timer will be reset for the next event
        """
    if len(self.pressed) != 1:
        return
    for key in self.pressed:
        self.pressed[key] = time.time() - self.repeat_delay + self.repeat_next
        return