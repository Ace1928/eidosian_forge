import numpy as np
from numpy.testing import assert_allclose
import pytest
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_axvspan():
    ax = plt.subplot(projection='polar')
    span = ax.axvspan(0, np.pi / 4)
    assert span.get_path()._interpolation_steps > 1