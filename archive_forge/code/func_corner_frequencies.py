from __future__ import absolute_import, division, print_function
import numpy as np
from ..processors import Processor
@property
def corner_frequencies(self):
    """Corner frequencies of the filter bands."""
    freqs = []
    for band in range(self.num_bands):
        bins = np.nonzero(self[:, band])[0]
        freqs.append([np.min(bins), np.max(bins)])
    return bins2frequencies(freqs, self.bin_frequencies)