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
@check_figures_equal(extensions=['pdf'])
@pytest.mark.parametrize('texsystem', ('pdflatex', 'xelatex', 'lualatex'))
@pytest.mark.backend('pgf')
def test_minus_signs_with_tex(fig_test, fig_ref, texsystem):
    if not _check_for_pgf(texsystem):
        pytest.skip(texsystem + ' + pgf is required')
    mpl.rcParams['pgf.texsystem'] = texsystem
    fig_test.text(0.5, 0.5, '$-1$')
    fig_ref.text(0.5, 0.5, '$−1$')