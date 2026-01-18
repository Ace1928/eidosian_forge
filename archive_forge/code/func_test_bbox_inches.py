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
@mpl.style.context('default')
@pytest.mark.backend('pgf')
def test_bbox_inches():
    mpl.rcParams.update({'font.family': 'serif', 'pgf.rcfonts': False})
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(range(5))
    ax2.plot(range(5))
    plt.tight_layout()
    bbox = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    compare_figure('pgf_bbox_inches.pdf', savefig_kwargs={'bbox_inches': bbox}, tol=0)