import base64
import io
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest
from matplotlib.testing.decorators import (
import matplotlib.pyplot as plt
from matplotlib import patches, transforms
from matplotlib.path import Path
def test_clipping_full():
    p = Path([[1e+30, 1e+30]] * 5)
    simplified = list(p.iter_segments(clip=[0, 0, 100, 100]))
    assert simplified == []
    p = Path([[50, 40], [75, 65]], [1, 2])
    simplified = list(p.iter_segments(clip=[0, 0, 100, 100]))
    assert [(list(x), y) for x, y in simplified] == [([50, 40], 1), ([75, 65], 2)]
    p = Path([[50, 40]], [1])
    simplified = list(p.iter_segments(clip=[0, 0, 100, 100]))
    assert [(list(x), y) for x, y in simplified] == [([50, 40], 1)]