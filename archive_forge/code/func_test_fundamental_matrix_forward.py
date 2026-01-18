import re
import textwrap
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from skimage.transform import (
from skimage.transform._geometric import (
@pytest.mark.parametrize('array_like_input', [False, True])
def test_fundamental_matrix_forward(array_like_input):
    if array_like_input:
        rotation = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        translation = (1, 0, 0)
    else:
        rotation = np.eye(3)
        translation = np.array([1, 0, 0])
    essential_matrix_tform = EssentialMatrixTransform(rotation=rotation, translation=translation)
    if array_like_input:
        params = [list(p) for p in essential_matrix_tform.params]
    else:
        params = essential_matrix_tform.params
    tform = FundamentalMatrixTransform(matrix=params)
    src = np.array([[0, 0], [0, 1], [1, 1]])
    assert_almost_equal(tform(src), [[0, -1, 0], [0, -1, 1], [0, -1, 1]])