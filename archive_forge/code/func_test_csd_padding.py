from numpy.testing import (assert_allclose, assert_almost_equal,
import numpy as np
import pytest
from matplotlib import mlab
def test_csd_padding(self):
    """Test zero padding of csd()."""
    if self.NFFT_density is None:
        return
    sargs = dict(x=self.y, y=self.y + 1, Fs=self.Fs, window=mlab.window_none, sides=self.sides)
    spec0, _ = mlab.csd(NFFT=self.NFFT_density, **sargs)
    spec1, _ = mlab.csd(NFFT=self.NFFT_density * 2, **sargs)
    assert_almost_equal(np.sum(np.conjugate(spec0) * spec0).real, np.sum(np.conjugate(spec1 / 2) * spec1 / 2).real)