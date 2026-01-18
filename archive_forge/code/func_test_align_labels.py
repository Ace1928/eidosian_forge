import gc
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import gridspec, ticker
def test_align_labels():
    """
    Tests for a bug in which constrained layout and align_ylabels on
    three unevenly sized subplots, one of whose y tick labels include
    negative numbers, drives the non-negative subplots' y labels off
    the edge of the plot
    """
    fig, (ax3, ax1, ax2) = plt.subplots(3, 1, layout='constrained', figsize=(6.4, 8), gridspec_kw={'height_ratios': (1, 1, 0.7)})
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Label')
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_ylabel('Label')
    ax3.set_ylim(0, 1)
    ax3.set_ylabel('Label')
    fig.align_ylabels(axs=(ax3, ax1, ax2))
    fig.draw_without_rendering()
    after_align = [ax1.yaxis.label.get_window_extent(), ax2.yaxis.label.get_window_extent(), ax3.yaxis.label.get_window_extent()]
    np.testing.assert_allclose([after_align[0].x0, after_align[2].x0], after_align[1].x0, rtol=0, atol=1e-05)
    assert after_align[0].x0 >= 1