import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import ndimage
from skimage.filters import median, rank
from skimage._shared.testing import assert_stacklevel
@pytest.mark.parametrize('mode, cval, behavior, warning_type', [('nearest', 0.0, 'ndimage', None), ('constant', 0.0, 'rank', UserWarning), ('nearest', 0.0, 'rank', None), ('nearest', 0.0, 'ndimage', None)])
def test_median_warning(image, mode, cval, behavior, warning_type):
    if warning_type:
        with pytest.warns(warning_type) as record:
            median(image, mode=mode, behavior=behavior)
        assert_stacklevel(record)
    else:
        median(image, mode=mode, behavior=behavior)