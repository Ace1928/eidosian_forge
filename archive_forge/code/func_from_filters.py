from __future__ import absolute_import, division, print_function
import numpy as np
from ..processors import Processor
@classmethod
def from_filters(cls, filters, bin_frequencies):
    """
        Create a filterbank with possibly multiple filters per band.

        Parameters
        ----------
        filters : list (of lists) of Filters
            List of Filters (per band); if multiple filters per band are
            desired, they should be also contained in a list, resulting in a
            list of lists of Filters.
        bin_frequencies : numpy array
            Frequencies of the bins (needed to determine the expected size of
            the filterbank).

        Returns
        -------
        filterbank : :class:`Filterbank` instance
            Filterbank with respective filter elements.

        """
    fb = np.zeros((len(bin_frequencies), len(filters)))
    for band_id, band_filter in enumerate(filters):
        band = fb[:, band_id]
        if isinstance(band_filter, list):
            for filt in band_filter:
                cls._put_filter(filt, band)
        else:
            cls._put_filter(band_filter, band)
    return Filterbank.__new__(cls, fb, bin_frequencies)