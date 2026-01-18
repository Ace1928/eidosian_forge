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
def test_legend_title_fontprop_fontsize():
    plt.plot(range(10))
    with pytest.raises(ValueError):
        plt.legend(title='Aardvark', title_fontsize=22, title_fontproperties={'family': 'serif', 'size': 22})
    leg = plt.legend(title='Aardvark', title_fontproperties=FontProperties(family='serif', size=22))
    assert leg.get_title().get_size() == 22
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes = axes.flat
    axes[0].plot(range(10))
    leg0 = axes[0].legend(title='Aardvark', title_fontsize=22)
    assert leg0.get_title().get_fontsize() == 22
    axes[1].plot(range(10))
    leg1 = axes[1].legend(title='Aardvark', title_fontproperties={'family': 'serif', 'size': 22})
    assert leg1.get_title().get_fontsize() == 22
    axes[2].plot(range(10))
    mpl.rcParams['legend.title_fontsize'] = None
    leg2 = axes[2].legend(title='Aardvark', title_fontproperties={'family': 'serif'})
    assert leg2.get_title().get_fontsize() == mpl.rcParams['font.size']
    axes[3].plot(range(10))
    leg3 = axes[3].legend(title='Aardvark')
    assert leg3.get_title().get_fontsize() == mpl.rcParams['font.size']
    axes[4].plot(range(10))
    mpl.rcParams['legend.title_fontsize'] = 20
    leg4 = axes[4].legend(title='Aardvark', title_fontproperties={'family': 'serif'})
    assert leg4.get_title().get_fontsize() == 20
    axes[5].plot(range(10))
    leg5 = axes[5].legend(title='Aardvark')
    assert leg5.get_title().get_fontsize() == 20