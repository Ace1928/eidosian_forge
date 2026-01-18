import warnings
from numpy.testing import assert_equal, assert_almost_equal
import os
import sys
import numpy as np
import skvideo.io
import skvideo.datasets
import skvideo.measure
@unittest.skip('Disabled pending BLAS check')
def test_measure_STRRED():
    vidpaths = skvideo.datasets.fullreferencepair()
    ref = skvideo.io.vread(vidpaths[0], as_grey=True)[:12]
    dis = skvideo.io.vread(vidpaths[1], as_grey=True)[:12]
    strred_array, strred, strredssn = skvideo.measure.strred(ref, dis)
    expected_array = np.array([[7.973215579986572, 21.01338768005371, 1.136332869529724, 3.105512380599976], [7.190542221069336, 28.211503982543945, 0.824764370918274, 11.768671989440918], [7.762616157531738, 30.080577850341797, 0.483192980289459, 11.239683151245117], [7.838700771331787, 29.70119285583496, 0.275575548410416, 1.08821713924408], [6.29062032699585, 31.81264877319336, 0.417621076107025, 11.035059928894043], [7.427119731903076, 23.272958755493164, 0.656901776790619, 0.641671419143677]])
    for j in range(6):
        for i in range(4):
            assert_almost_equal(strred_array[j, i], expected_array[j, i], decimal=3)
    assert_almost_equal(strred, 202.75794982910156, decimal=3)
    assert_almost_equal(strredssn, 4.097815036773682, decimal=3)