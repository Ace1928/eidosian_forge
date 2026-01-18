import re
import textwrap
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from skimage.transform import (
from skimage.transform._geometric import (
def test_essential_matrix_init():
    tform = EssentialMatrixTransform(rotation=np.eye(3), translation=np.array([0, 0, 1]))
    assert_equal(tform.params, np.array([0, -1, 0, 1, 0, 0, 0, 0, 0]).reshape(3, 3))