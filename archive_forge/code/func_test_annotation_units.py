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
@check_figures_equal(extensions=['png'])
def test_annotation_units(fig_test, fig_ref):
    ax = fig_test.add_subplot()
    ax.plot(datetime.now(), 1, 'o')
    ax.annotate('x', (datetime.now(), 0.5), xycoords=('data', 'axes fraction'), xytext=(0, 0), textcoords='offset points')
    ax = fig_ref.add_subplot()
    ax.plot(datetime.now(), 1, 'o')
    ax.annotate('x', (datetime.now(), 0.5), xycoords=('data', 'axes fraction'))