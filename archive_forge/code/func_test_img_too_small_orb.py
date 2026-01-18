import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal
from skimage import data
from skimage._shared.testing import run_in_parallel, xfail, arch32
from skimage.feature import ORB
from skimage.util.dtype import _convert
def test_img_too_small_orb():
    img = data.brick()[:64, :64]
    detector_extractor = ORB(downscale=2, n_scales=8)
    detector_extractor.detect(img)
    detector_extractor.detect_and_extract(img)