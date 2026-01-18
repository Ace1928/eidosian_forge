from numpy.testing import (assert_allclose, assert_almost_equal,
import numpy as np
import pytest
from matplotlib import mlab
def check_maxfreq(self, spec, fsp, fstims):
    if len(fstims) == 0:
        return
    if fsp.min() < 0:
        fspa = np.abs(fsp)
        zeroind = fspa.argmin()
        self.check_maxfreq(spec[:zeroind], fspa[:zeroind], fstims)
        self.check_maxfreq(spec[zeroind:], fspa[zeroind:], fstims)
        return
    fstimst = fstims[:]
    spect = spec.copy()
    while fstimst:
        maxind = spect.argmax()
        maxfreq = fsp[maxind]
        assert_almost_equal(maxfreq, fstimst[-1])
        del fstimst[-1]
        spect[maxind - 5:maxind + 5] = 0