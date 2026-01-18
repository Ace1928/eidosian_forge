import numpy as np
from numpy.testing import assert_equal
from skimage._shared import version_requirements as version_req
from skimage._shared import testing
def test_is_installed():
    assert version_req.is_installed('python', '>=2.7')
    assert not version_req.is_installed('numpy', '<1.0')