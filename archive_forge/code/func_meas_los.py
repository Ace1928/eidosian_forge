from __future__ import annotations
import numpy as np
from .channels import DriveChannel, MeasureChannel
from .exceptions import PulseError
@property
def meas_los(self) -> dict[MeasureChannel, float]:
    """Returns dictionary mapping measure channels (MeasureChannel) to los."""
    return self._m_lo_freq