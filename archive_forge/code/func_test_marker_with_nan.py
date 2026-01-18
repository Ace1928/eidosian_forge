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
def test_marker_with_nan():
    fig, ax = plt.subplots(1)
    steps = 1000
    data = np.arange(steps)
    ax.semilogx(data)
    ax.fill_between(data, data * 0.8, data * 1.2)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')