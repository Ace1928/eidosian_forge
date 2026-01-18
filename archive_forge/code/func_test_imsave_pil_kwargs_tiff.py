from contextlib import ExitStack
from copy import copy
import functools
import io
import os
from pathlib import Path
import platform
import sys
import urllib.request
import numpy as np
from numpy.testing import assert_array_equal
from PIL import Image
import matplotlib as mpl
from matplotlib import (
from matplotlib.image import (AxesImage, BboxImage, FigureImage,
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.transforms import Bbox, Affine2D, TransformedBbox
import matplotlib.ticker as mticker
import pytest
def test_imsave_pil_kwargs_tiff():
    from PIL.TiffTags import TAGS_V2 as TAGS
    buf = io.BytesIO()
    pil_kwargs = {'description': 'test image'}
    plt.imsave(buf, [[0, 1], [2, 3]], format='tiff', pil_kwargs=pil_kwargs)
    assert len(pil_kwargs) == 1
    im = Image.open(buf)
    tags = {TAGS[k].name: v for k, v in im.tag_v2.items()}
    assert tags['ImageDescription'] == 'test image'