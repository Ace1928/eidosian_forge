import re
import sys
import numpy as np
import pytest
from matplotlib import _preprocess_data
from matplotlib.axes import Axes
from matplotlib.testing import subprocess_run_for_testing
from matplotlib.testing.decorators import check_figures_equal
@pytest.mark.parametrize('plotter', plotters)
@check_figures_equal(extensions=['png'])
def test_dict_unpack(self, plotter, fig_test, fig_ref):
    x = [1, 2, 3]
    y = [4, 5, 6]
    ddict = dict(zip(x, y))
    plotter(fig_test.subplots(), ddict.keys(), ddict.values())
    plotter(fig_ref.subplots(), x, y)