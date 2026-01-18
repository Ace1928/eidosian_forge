import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_label_output_wrong_size():
    data = np.ones([5])
    for t in types:
        output = np.zeros([10], t)
        assert_raises((RuntimeError, ValueError), ndimage.label, data, output=output)