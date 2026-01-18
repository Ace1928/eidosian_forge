import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal
from skimage import data
from skimage._shared.testing import run_in_parallel, xfail, arch32
from skimage.feature import ORB
from skimage.util.dtype import _convert
def test_no_descriptors_extracted_orb():
    img = np.ones((128, 128))
    detector_extractor = ORB()
    with pytest.raises(RuntimeError):
        detector_extractor.detect_and_extract(img)