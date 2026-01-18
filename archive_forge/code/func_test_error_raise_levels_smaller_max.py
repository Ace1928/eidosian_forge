import numpy as np
import pytest
from skimage._shared.testing import expected_warnings, run_in_parallel
from skimage.feature import (
from skimage.transform import integral_image
def test_error_raise_levels_smaller_max(self):
    with pytest.raises(ValueError):
        graycomatrix(self.image - 1, [1], [np.pi], 3)