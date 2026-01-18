from __future__ import annotations
import numpy as np
from .channels import DriveChannel, MeasureChannel
from .exceptions import PulseError
@property
def qubit_los(self) -> dict[DriveChannel, float]:
    """Returns dictionary mapping qubit channels (DriveChannel) to los."""
    return self._q_lo_freq