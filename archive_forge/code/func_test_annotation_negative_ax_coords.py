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
@image_comparison(['annotation_negative_ax_coords.png'])
def test_annotation_negative_ax_coords():
    fig, ax = plt.subplots()
    ax.annotate('+ pts', xytext=[30, 20], textcoords='axes points', xy=[30, 20], xycoords='axes points', fontsize=32)
    ax.annotate('- pts', xytext=[30, -20], textcoords='axes points', xy=[30, -20], xycoords='axes points', fontsize=32, va='top')
    ax.annotate('+ frac', xytext=[0.75, 0.05], textcoords='axes fraction', xy=[0.75, 0.05], xycoords='axes fraction', fontsize=32)
    ax.annotate('- frac', xytext=[0.75, -0.05], textcoords='axes fraction', xy=[0.75, -0.05], xycoords='axes fraction', fontsize=32, va='top')
    ax.annotate('+ pixels', xytext=[160, 25], textcoords='axes pixels', xy=[160, 25], xycoords='axes pixels', fontsize=32)
    ax.annotate('- pixels', xytext=[160, -25], textcoords='axes pixels', xy=[160, -25], xycoords='axes pixels', fontsize=32, va='top')