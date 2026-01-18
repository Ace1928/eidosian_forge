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
@image_comparison(['large_subscript_title.png'], style='mpl20')
def test_large_subscript_title():
    plt.rcParams['text.kerning_factor'] = 6
    plt.rcParams['axes.titley'] = None
    fig, axs = plt.subplots(1, 2, figsize=(9, 2.5), constrained_layout=True)
    ax = axs[0]
    ax.set_title('$\\sum_{i} x_i$')
    ax.set_title('New way', loc='left')
    ax.set_xticklabels([])
    ax = axs[1]
    ax.set_title('$\\sum_{i} x_i$', y=1.01)
    ax.set_title('Old Way', loc='left')
    ax.set_xticklabels([])