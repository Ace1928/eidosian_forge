from __future__ import absolute_import, division, print_function
import sys
import numpy as np
from ..audio.signal import smooth as smooth_signal
from ..processors import BufferProcessor, OnlineProcessor
@property
def min_interval(self):
    """Minimum beat interval [frames]."""
    return self.histogram_processor.min_interval