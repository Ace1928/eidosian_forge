from collections import Counter
from pathlib import Path
import io
import re
import tempfile
import numpy as np
import pytest
from matplotlib import cbook, path, patheffects, font_manager as fm
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from matplotlib.testing._markers import needs_ghostscript, needs_usetex
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import matplotlib as mpl
import matplotlib.collections as mcollections
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
def test_auto_papersize_deprecation():
    fig = plt.figure()
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        fig.savefig(io.BytesIO(), format='eps', papertype='auto')
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        mpl.rcParams['ps.papersize'] = 'auto'