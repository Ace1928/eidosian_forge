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
@pytest.mark.parametrize('text', ['', 'O'], ids=['empty', 'non-empty'])
def test_non_default_dpi(text):
    fig, ax = plt.subplots()
    t1 = ax.text(0.5, 0.5, text, ha='left', va='bottom')
    fig.canvas.draw()
    dpi = fig.dpi
    bbox1 = t1.get_window_extent()
    bbox2 = t1.get_window_extent(dpi=dpi * 10)
    np.testing.assert_allclose(bbox2.get_points(), bbox1.get_points() * 10, rtol=0.05)
    assert fig.dpi == dpi