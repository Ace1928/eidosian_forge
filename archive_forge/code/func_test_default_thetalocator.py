import numpy as np
from numpy.testing import assert_allclose
import pytest
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_default_thetalocator():
    fig, axs = plt.subplot_mosaic('AAAABB.', subplot_kw={'projection': 'polar'})
    for ax in axs.values():
        ax.set_thetalim(0, np.pi)
    for ax in axs.values():
        ticklocs = np.degrees(ax.xaxis.get_majorticklocs()).tolist()
        assert pytest.approx(90) in ticklocs
        assert pytest.approx(100) not in ticklocs