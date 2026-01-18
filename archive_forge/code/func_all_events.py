from __future__ import annotations
from typing import (
from simpy.exceptions import Interrupt
@staticmethod
def all_events(events: Tuple[Event, ...], count: int) -> bool:
    """An evaluation function that returns ``True`` if all *events* have
        been triggered."""
    return len(events) == count