import warnings
from numpy.testing import assert_equal, assert_almost_equal
import os
import sys
import numpy as np
import skvideo.io
import skvideo.datasets
import skvideo.measure
def test_measure_Viideo():
    vidpaths = skvideo.datasets.bigbuckbunny()
    dis = skvideo.io.vread(vidpaths, as_grey=True)
    dis = dis[:80, :200, :200]
    score = skvideo.measure.viideo_score(dis)
    assert_almost_equal(score, 0.72179317, decimal=8)