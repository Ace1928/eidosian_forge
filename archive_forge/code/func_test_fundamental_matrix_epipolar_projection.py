import re
import textwrap
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from skimage.transform import (
from skimage.transform._geometric import (
def test_fundamental_matrix_epipolar_projection():
    src = np.array([1.839035, 1.924743, 0.543582, 0.375221, 0.47324, 0.142522, 0.96491, 0.598376, 0.102388, 0.140092, 15.994343, 9.622164, 0.285901, 0.430055, 0.09115, 0.254594]).reshape(-1, 2)
    dst = np.array([1.002114, 1.129644, 1.521742, 1.846002, 1.084332, 0.275134, 0.293328, 0.588992, 0.839509, 0.08729, 1.779735, 1.116857, 0.878616, 0.602447, 0.642616, 1.028681]).reshape(-1, 2)
    tform = estimate_transform('fundamental', src, dst)
    p = np.abs(np.sum(np.column_stack((dst, np.ones(len(dst)))) * tform(src), axis=1))
    assert np.all(p < 0.01)