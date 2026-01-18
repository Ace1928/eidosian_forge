from __future__ import annotations
from typing import (
from simpy.exceptions import Interrupt
@staticmethod
def any_events(events: Tuple[Event, ...], count: int) -> bool:
    """An evaluation function that returns ``True`` if at least one of
        *events* has been triggered."""
    return count > 0 or len(events) == 0