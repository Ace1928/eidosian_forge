import re
import textwrap
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from skimage.transform import (
from skimage.transform._geometric import (
def test_fundamental_matrix_inverse():
    essential_matrix_tform = EssentialMatrixTransform(rotation=np.eye(3), translation=np.array([1, 0, 0]))
    tform = FundamentalMatrixTransform()
    tform.params = essential_matrix_tform.params
    src = np.array([[0, 0], [0, 1], [1, 1]])
    assert_almost_equal(tform.inverse(src), [[0, 1, 0], [0, 1, -1], [0, 1, -1]])