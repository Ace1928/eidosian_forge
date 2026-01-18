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
@pytest.mark.backend('Agg')
def test_jpeg_dpi():
    plt.plot([0, 1, 2], [0, 1, 0])
    buf = io.BytesIO()
    plt.savefig(buf, format='jpg', dpi=200)
    im = Image.open(buf)
    assert im.info['dpi'] == (200, 200)