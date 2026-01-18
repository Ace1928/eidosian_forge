import numpy as np
import pytest
from scipy import ndimage
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.image import (
def test_img_to_graph_sparse():
    mask = np.zeros((2, 3), dtype=bool)
    mask[0, 0] = 1
    mask[:, 2] = 1
    x = np.zeros((2, 3))
    x[0, 0] = 1
    x[0, 2] = -1
    x[1, 2] = -2
    grad_x = img_to_graph(x, mask=mask).todense()
    desired = np.array([[1, 0, 0], [0, -1, 1], [0, 1, -2]])
    np.testing.assert_array_equal(grad_x, desired)