from datetime import datetime
import io
import warnings
import numpy as np
from numpy.testing import assert_almost_equal
from packaging.version import parse as parse_version
import pyparsing
import pytest
import matplotlib as mpl
from matplotlib.backend_bases import MouseEvent
from matplotlib.font_manager import FontProperties
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.testing._markers import needs_usetex
from matplotlib.text import Text, Annotation, OffsetFrom
def test_annotation_update():
    fig, ax = plt.subplots(1, 1)
    an = ax.annotate('annotation', xy=(0.5, 0.5))
    extent1 = an.get_window_extent(fig.canvas.get_renderer())
    fig.tight_layout()
    extent2 = an.get_window_extent(fig.canvas.get_renderer())
    assert not np.allclose(extent1.get_points(), extent2.get_points(), rtol=1e-06)