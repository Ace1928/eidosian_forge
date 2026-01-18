import io
import numpy as np
from numpy.testing import assert_array_almost_equal
from PIL import Image, TiffTags
import pytest
from matplotlib import (
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.figure import Figure
from matplotlib.image import imread
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison
from matplotlib.transforms import IdentityTransform
def test_long_path():
    buff = io.BytesIO()
    fig = Figure()
    ax = fig.subplots()
    points = np.ones(100000)
    points[::2] *= -1
    ax.plot(points)
    fig.savefig(buff, format='png')