import re
import textwrap
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from skimage.transform import (
from skimage.transform._geometric import (
def test_affine_estimation():
    tform = estimate_transform('affine', SRC[:3, :], DST[:3, :])
    assert_almost_equal(tform(SRC[:3, :]), DST[:3, :])
    tform2 = estimate_transform('affine', SRC, DST)
    assert_almost_equal(tform2.inverse(tform2(SRC)), SRC)
    tform3 = AffineTransform()
    assert tform3.estimate(SRC, DST)
    assert_almost_equal(tform3.params, tform2.params)