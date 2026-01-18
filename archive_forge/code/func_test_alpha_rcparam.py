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
@image_comparison(['rcparam_alpha.png'], remove_text=True, tol=0 if platform.machine() == 'x86_64' else 0.01)
def test_alpha_rcparam():
    fig, ax = plt.subplots()
    ax.plot(range(10), lw=5)
    with mpl.rc_context(rc={'legend.framealpha': 0.75}):
        leg = plt.legend(['Longlabel that will go away'], loc='center')
        leg.legendPatch.set_facecolor([1, 0, 0, 0.5])