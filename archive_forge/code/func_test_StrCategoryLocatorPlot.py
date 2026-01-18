import warnings
import pytest
import numpy as np
import matplotlib as mpl
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.category as cat
from matplotlib.testing.decorators import check_figures_equal
@pytest.mark.parametrize('plotter', PLOT_LIST, ids=PLOT_IDS)
def test_StrCategoryLocatorPlot(self, plotter):
    ax = plt.figure().subplots()
    plotter(ax, [1, 2, 3], ['a', 'b', 'c'])
    np.testing.assert_array_equal(ax.yaxis.major.locator(), range(3))