import itertools
import numpy as np
import pytest
from matplotlib.axes import Axes, SubplotBase
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal, image_comparison
def test_get_gridspec():
    fig, ax = plt.subplots()
    assert ax.get_subplotspec().get_gridspec() == ax.get_gridspec()