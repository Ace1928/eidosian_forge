import base64
import io
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest
from matplotlib.testing.decorators import (
import matplotlib.pyplot as plt
from matplotlib import patches, transforms
from matplotlib.path import Path
@image_comparison(['clipping_with_nans'])
def test_clipping_with_nans():
    x = np.linspace(0, 3.14 * 2, 3000)
    y = np.sin(x)
    x[::100] = np.nan
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_ylim(-0.25, 0.25)