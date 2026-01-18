from __future__ import absolute_import, division, print_function
import numpy as np
from ..processors import Processor
@property
def center_frequencies(self):
    """Center frequencies of the filter bands."""
    freqs = []
    for band in range(self.num_bands):
        bins = np.nonzero(self[:, band])[0]
        min_bin = np.min(bins)
        max_bin = np.max(bins)
        if self[min_bin, band] == self[max_bin, band]:
            center = int(min_bin + (max_bin - min_bin) / 2.0)
        else:
            center = min_bin + np.argmax(self[min_bin:max_bin, band])
        freqs.append(center)
    return bins2frequencies(freqs, self.bin_frequencies)