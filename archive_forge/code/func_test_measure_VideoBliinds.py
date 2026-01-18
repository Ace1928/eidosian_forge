import warnings
from numpy.testing import assert_equal, assert_almost_equal
import os
import sys
import numpy as np
import skvideo.io
import skvideo.datasets
import skvideo.measure
def test_measure_VideoBliinds():
    vidpaths = skvideo.datasets.bigbuckbunny()
    dis = skvideo.io.vread(vidpaths, as_grey=True)
    dis = dis[:20, :200, :200]
    features = skvideo.measure.videobliinds_features(dis)
    output = np.array([2.5088, 0.724275349118, 0.88615, 0.094900845105, 0.075270971672, 0.153258614032, 0.83295, 0.145853236056, 0.047469409576, 0.156152160875, 0.85815, 0.043908992567, 0.093810932519, 0.128847087445, 0.857375, 0.07021082743, 0.083866251922, 0.138073680721, 3.172, 0.970150022342, 1.004175, 0.040410359225, 0.212755111969, 0.253072861341, 0.961, 0.186445182929, 0.124771296044, 0.300422186113, 0.8754, -0.018816730259, 0.206918460682, 0.191991528548, 0.901925, 0.030521899481, 0.192355885179, 0.215250702742, 9.489535449185, 1.009615778923, 0.250374922528, 0.713554522195, 0.67227717825, 0.669912498498, 0.692061577463, 0.621884675219, 0.441264439346, 0.142519297382])
    for i in range(features.shape[0]):
        assert_almost_equal(features[i], output[i], decimal=10)