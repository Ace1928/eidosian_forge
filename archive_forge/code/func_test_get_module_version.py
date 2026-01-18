import numpy as np
from numpy.testing import assert_equal
from skimage._shared import version_requirements as version_req
from skimage._shared import testing
def test_get_module_version():
    assert version_req.get_module_version('numpy')
    assert version_req.get_module_version('scipy')
    with testing.raises(ImportError):
        version_req.get_module_version('fakenumpy')