import warnings
import pytest
import numpy as np
import matplotlib as mpl
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.category as cat
from matplotlib.testing.decorators import check_figures_equal
def test_no_deprecation_on_empty_data():
    """
    Smoke test to check that no deprecation warning is emitted. See #22640.
    """
    f, ax = plt.subplots()
    ax.xaxis.update_units(['a', 'b'])
    ax.plot([], [])