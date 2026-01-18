import datetime
from io import BytesIO
import os
import shutil
import numpy as np
from packaging.version import parse as parse_version
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.testing import _has_tex_package, _check_for_pgf
from matplotlib.testing.exceptions import ImageComparisonFailure
from matplotlib.testing.compare import compare_images
from matplotlib.backends.backend_pgf import PdfPages
from matplotlib.testing.decorators import (
from matplotlib.testing._markers import (
@needs_pgf_xelatex
@pytest.mark.backend('pgf')
@image_comparison(['pgf_mixedmode.pdf'], style='default')
def test_mixedmode():
    mpl.rcParams.update({'font.family': 'serif', 'pgf.rcfonts': False})
    Y, X = np.ogrid[-1:1:40j, -1:1:40j]
    plt.pcolor(X ** 2 + Y ** 2).set_rasterized(True)