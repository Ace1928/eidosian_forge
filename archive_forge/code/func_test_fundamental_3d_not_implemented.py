import re
import textwrap
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from skimage.transform import (
from skimage.transform._geometric import (
def test_fundamental_3d_not_implemented():
    with pytest.raises(NotImplementedError):
        _ = FundamentalMatrixTransform(dimensionality=3)
    with pytest.raises(NotImplementedError):
        _ = FundamentalMatrixTransform(np.eye(4))