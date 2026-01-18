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
@image_comparison(['annotation_negative_fig_coords.png'])
def test_annotation_negative_fig_coords():
    fig, ax = plt.subplots()
    ax.annotate('+ pts', xytext=[10, 120], textcoords='figure points', xy=[10, 120], xycoords='figure points', fontsize=32)
    ax.annotate('- pts', xytext=[-10, 180], textcoords='figure points', xy=[-10, 180], xycoords='figure points', fontsize=32, va='top')
    ax.annotate('+ frac', xytext=[0.05, 0.55], textcoords='figure fraction', xy=[0.05, 0.55], xycoords='figure fraction', fontsize=32)
    ax.annotate('- frac', xytext=[-0.05, 0.5], textcoords='figure fraction', xy=[-0.05, 0.5], xycoords='figure fraction', fontsize=32, va='top')
    ax.annotate('+ pixels', xytext=[50, 50], textcoords='figure pixels', xy=[50, 50], xycoords='figure pixels', fontsize=32)
    ax.annotate('- pixels', xytext=[-50, 100], textcoords='figure pixels', xy=[-50, 100], xycoords='figure pixels', fontsize=32, va='top')