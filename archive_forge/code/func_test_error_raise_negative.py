import numpy as np
import pytest
from skimage._shared.testing import expected_warnings, run_in_parallel
from skimage.feature import (
from skimage.transform import integral_image
def test_error_raise_negative(self):
    with pytest.raises(ValueError):
        graycomatrix(self.image.astype(np.int16) - 1, [1], [np.pi], 4)