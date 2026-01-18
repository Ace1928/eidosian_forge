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
def test_bbox_inches_tight():
    fig, ax = plt.subplots()
    ax.imshow([[0, 1], [2, 3]])
    fig.savefig(BytesIO(), format='pdf', backend='pgf', bbox_inches='tight')