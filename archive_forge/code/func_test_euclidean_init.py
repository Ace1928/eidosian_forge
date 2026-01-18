import re
import textwrap
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from skimage.transform import (
from skimage.transform._geometric import (
def test_euclidean_init():
    rotation = 1
    translation = (1, 1)
    tform = EuclideanTransform(rotation=rotation, translation=translation)
    assert_almost_equal(tform.rotation, rotation)
    assert_almost_equal(tform.translation, translation)
    tform2 = EuclideanTransform(tform.params)
    assert_almost_equal(tform2.rotation, rotation)
    assert_almost_equal(tform2.translation, translation)
    rotation = 0
    translation = (1, 1)
    tform = EuclideanTransform(rotation=rotation, translation=translation)
    assert_almost_equal(tform.rotation, rotation)
    assert_almost_equal(tform.translation, translation)
    rotation = np.pi / 2
    translation = (1, 1)
    tform = EuclideanTransform(rotation=rotation, translation=translation)
    assert_almost_equal(tform.rotation, rotation)
    assert_almost_equal(tform.translation, translation)