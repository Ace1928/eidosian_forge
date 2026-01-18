from __future__ import annotations
from typing import (
from simpy.exceptions import Interrupt
@defused.setter
def defused(self, value: bool) -> None:
    self._defused = True