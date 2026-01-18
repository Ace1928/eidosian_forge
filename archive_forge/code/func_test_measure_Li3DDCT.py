import warnings
from numpy.testing import assert_equal, assert_almost_equal
import os
import sys
import numpy as np
import skvideo.io
import skvideo.datasets
import skvideo.measure
def test_measure_Li3DDCT():
    vidpaths = skvideo.datasets.fullreferencepair()
    vidpaths = skvideo.datasets.bigbuckbunny()
    dis = skvideo.io.vread(vidpaths, as_grey=True)
    dis = dis[:10, :200, :200]
    Li_array = skvideo.measure.Li3DDCT_features(dis)
    print(Li_array)