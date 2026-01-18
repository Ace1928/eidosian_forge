from __future__ import absolute_import, division, print_function
import sys
import numpy as np
from ..audio.signal import smooth as smooth_signal
from ..processors import BufferProcessor, OnlineProcessor
def interval_histogram(self, activations, **kwargs):
    """
        Compute the histogram of the beat intervals.

        Parameters
        ----------
        activations : numpy array
            Beat activation function.

        Returns
        -------
        histogram_bins : numpy array
            Bins of the beat interval histogram.
        histogram_delays : numpy array
            Corresponding delays [frames].

        """
    return self.histogram_processor(activations, **kwargs)