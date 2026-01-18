import warnings
from numpy.testing import assert_equal, assert_almost_equal
import os
import sys
import numpy as np
import skvideo.io
import skvideo.datasets
import skvideo.measure
def test_measure_MAE():
    vidpaths = skvideo.datasets.fullreferencepair()
    ref = skvideo.io.vread(vidpaths[0], as_grey=True)
    dis = skvideo.io.vread(vidpaths[1], as_grey=True)
    scores = skvideo.measure.mae(ref, dis)
    avg_score = np.mean(scores)
    assert_almost_equal(avg_score, 11.515880584716797, decimal=15)