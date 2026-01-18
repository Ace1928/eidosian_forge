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
@image_comparison(['agg_text_clip.png'])
def test_agg_text_clip():
    np.random.seed(1)
    fig, (ax1, ax2) = plt.subplots(2)
    for x, y in np.random.rand(10, 2):
        ax1.text(x, y, 'foo', clip_on=True)
        ax2.text(x, y, 'foo')