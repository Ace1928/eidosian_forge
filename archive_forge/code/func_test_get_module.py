import numpy as np
from numpy.testing import assert_equal
from skimage._shared import version_requirements as version_req
from skimage._shared import testing
def test_get_module():
    assert version_req.get_module('numpy') is np