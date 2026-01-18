import re
import textwrap
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from skimage.transform import (
from skimage.transform._geometric import (
def test_matrix_transform():
    tform = AffineTransform(scale=(0.1, 0.5), rotation=2)
    assert_equal(tform(SRC), matrix_transform(SRC, tform.params))