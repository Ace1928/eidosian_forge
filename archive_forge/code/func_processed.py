from __future__ import annotations
from typing import (
from simpy.exceptions import Interrupt
@property
def processed(self) -> bool:
    """Becomes ``True`` if the event has been processed (e.g., its
        callbacks have been invoked)."""
    return self.callbacks is None