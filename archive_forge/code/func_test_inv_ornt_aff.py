import warnings
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ..affines import from_matvec, to_matvec
from ..orientations import (
from ..testing import deprecated_to, expires
def test_inv_ornt_aff():
    with pytest.raises(OrientationError):
        inv_ornt_aff([[0, 1], [1, -1], [np.nan, np.nan]], (3, 4, 5))