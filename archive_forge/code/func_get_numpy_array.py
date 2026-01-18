import sys
import os
import struct
import logging
import numpy as np
def get_numpy_array(self):
    """Get (load) the data that this DicomSeries represents, and return
        it as a numpy array. If this serie contains multiple images, the
        resulting array is 3D, otherwise it's 2D.
        """
    if len(self) == 0:
        raise ValueError('Serie does not contain any files.')
    elif len(self) == 1:
        return self[0].get_numpy_array()
    if self.info is None:
        raise RuntimeError('Cannot return volume if series not finished.')
    slice = self[0].get_numpy_array()
    vol = np.zeros(self.shape, dtype=slice.dtype)
    vol[0] = slice
    self._progressIndicator.start('loading data', '', len(self))
    for z in range(1, len(self)):
        vol[z] = self[z].get_numpy_array()
        self._progressIndicator.set_progress(z + 1)
    self._progressIndicator.finish()
    import gc
    gc.collect()
    return vol