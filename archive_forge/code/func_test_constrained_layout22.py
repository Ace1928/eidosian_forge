import gc
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import gridspec, ticker
def test_constrained_layout22():
    """#11035: suptitle should not be include in CL if manually positioned"""
    fig, ax = plt.subplots(layout='constrained')
    fig.draw_without_rendering()
    extents0 = np.copy(ax.get_position().extents)
    fig.suptitle('Suptitle', y=0.5)
    fig.draw_without_rendering()
    extents1 = np.copy(ax.get_position().extents)
    np.testing.assert_allclose(extents0, extents1)