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
def test_hinting_factor_backends():
    plt.rcParams['text.hinting_factor'] = 1
    fig = plt.figure()
    t = fig.text(0.5, 0.5, 'some text')
    fig.savefig(io.BytesIO(), format='svg')
    expected = t.get_window_extent().intervalx
    fig.savefig(io.BytesIO(), format='png')
    np.testing.assert_allclose(t.get_window_extent().intervalx, expected, rtol=0.1)