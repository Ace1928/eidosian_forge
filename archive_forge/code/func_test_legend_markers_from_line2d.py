import collections
import platform
from unittest import mock
import warnings
import numpy as np
from numpy.testing import assert_allclose
import pytest
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.testing._markers import needs_usetex
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import matplotlib.collections as mcollections
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerTuple
import matplotlib.legend as mlegend
from matplotlib import _api, rc_context
from matplotlib.font_manager import FontProperties
def test_legend_markers_from_line2d():
    _markers = ['.', '*', 'v']
    fig, ax = plt.subplots()
    lines = [mlines.Line2D([0], [0], ls='None', marker=mark) for mark in _markers]
    labels = ['foo', 'bar', 'xyzzy']
    markers = [line.get_marker() for line in lines]
    legend = ax.legend(lines, labels)
    new_markers = [line.get_marker() for line in legend.get_lines()]
    new_labels = [text.get_text() for text in legend.get_texts()]
    assert markers == new_markers == _markers
    assert labels == new_labels