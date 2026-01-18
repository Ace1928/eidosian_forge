from __future__ import annotations
import numpy as np
from .channels import DriveChannel, MeasureChannel
from .exceptions import PulseError
def add_lo(self, channel: DriveChannel | MeasureChannel, freq: float):
    """Add a lo mapping for a channel."""
    if isinstance(channel, DriveChannel):
        self.check_lo(channel, freq)
        self._q_lo_freq[channel] = freq
    elif isinstance(channel, MeasureChannel):
        self.check_lo(channel, freq)
        self._m_lo_freq[channel] = freq
    else:
        raise PulseError('Specified channel %s cannot be configured.' % channel.name)