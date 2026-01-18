from __future__ import absolute_import, division, print_function
import warnings
import numpy as np
import scipy.fftpack as fftpack
from ..processors import Processor
from .signal import Signal, FramedSignal
@property
def num_bins(self):
    """Number of bins."""
    return int(self.shape[1])