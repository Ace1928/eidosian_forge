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
def test_loc_validation_string_value():
    fig, ax = plt.subplots()
    ax.legend(loc='best')
    ax.legend(loc='upper right')
    ax.legend(loc='best')
    ax.legend(loc='upper right')
    ax.legend(loc='upper left')
    ax.legend(loc='lower left')
    ax.legend(loc='lower right')
    ax.legend(loc='right')
    ax.legend(loc='center left')
    ax.legend(loc='center right')
    ax.legend(loc='lower center')
    ax.legend(loc='upper center')
    with pytest.raises(ValueError, match="'wrong' is not a valid value for"):
        ax.legend(loc='wrong')