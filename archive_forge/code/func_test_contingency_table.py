import numpy as np
import pytest
from skimage.metrics import (
from skimage._shared.testing import (
def test_contingency_table():
    im_true = np.array([1, 2, 3, 4])
    im_test = np.array([1, 1, 8, 8])
    table1 = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25]])
    sparse_table2 = contingency_table(im_true, im_test, normalize=True)
    table2 = sparse_table2.toarray()
    assert_array_equal(table1, table2)