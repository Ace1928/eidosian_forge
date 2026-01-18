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
def smooth2d(A, sigma=3):
    window_len = max(int(sigma), 3) * 2 + 1
    A = np.apply_along_axis(smooth1d, 0, A, window_len)
    A = np.apply_along_axis(smooth1d, 1, A, window_len)
    return A