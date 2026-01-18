import numpy as np
import pytest
from numpy.testing import assert_array_equal
from skimage.segmentation import (
def test_morphsnakes_3d():
    image = np.zeros((7, 7, 7))
    evolution = []

    def callback(x):
        evolution.append(x.sum())
    ls = morphological_chan_vese(image, 5, 'disk', iter_callback=callback)
    assert evolution[0] == 81
    assert ls.sum() == 0
    for v1, v2 in zip(evolution[:-1], evolution[1:]):
        assert v1 >= v2