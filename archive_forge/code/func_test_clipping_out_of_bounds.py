import base64
import io
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest
from matplotlib.testing.decorators import (
import matplotlib.pyplot as plt
from matplotlib import patches, transforms
from matplotlib.path import Path
def test_clipping_out_of_bounds():
    path = Path([(0, 0), (1, 2), (2, 1)])
    simplified = path.cleaned(clip=(10, 10, 20, 20))
    assert_array_equal(simplified.vertices, [(0, 0)])
    assert simplified.codes == [Path.STOP]
    path = Path([(0, 0), (1, 2), (2, 1)], [Path.MOVETO, Path.LINETO, Path.LINETO])
    simplified = path.cleaned(clip=(10, 10, 20, 20))
    assert_array_equal(simplified.vertices, [(0, 0)])
    assert simplified.codes == [Path.STOP]
    path = Path([(0, 0), (1, 2), (2, 3)], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
    simplified = path.cleaned()
    simplified_clipped = path.cleaned(clip=(10, 10, 20, 20))
    assert_array_equal(simplified.vertices, simplified_clipped.vertices)
    assert_array_equal(simplified.codes, simplified_clipped.codes)