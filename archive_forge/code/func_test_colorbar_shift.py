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
@image_comparison(['colorbar_shift.eps'], savefig_kwarg={'bbox_inches': 'tight'}, style='mpl20')
def test_colorbar_shift(tmp_path):
    cmap = mcolors.ListedColormap(['r', 'g', 'b'])
    norm = mcolors.BoundaryNorm([-1, -0.5, 0.5, 1], cmap.N)
    plt.scatter([0, 1], [1, 1], c=[0, 1], cmap=cmap, norm=norm)
    plt.colorbar()