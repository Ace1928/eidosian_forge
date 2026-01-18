import itertools
import numpy as np
import pytest
from matplotlib.axes import Axes, SubplotBase
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal, image_comparison
def test_label_outer_non_gridspec():
    ax = plt.axes((0, 0, 1, 1))
    ax.label_outer()
    check_ticklabel_visible([ax], [True], [True])