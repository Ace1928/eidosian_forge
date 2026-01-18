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
def test_legend_label_three_args(self):
    fig, ax = plt.subplots()
    lines = ax.plot(range(10))
    with pytest.raises(TypeError, match='0-2'):
        fig.legend(lines, ['foobar'], 'right')
    with pytest.raises(TypeError, match='0-2'):
        fig.legend(lines, ['foobar'], 'right', loc='left')