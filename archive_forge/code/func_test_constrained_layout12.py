import gc
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import gridspec, ticker
@image_comparison(['constrained_layout12.png'])
def test_constrained_layout12():
    """Test that very unbalanced labeling still works."""
    fig = plt.figure(layout='constrained', figsize=(6, 8))
    gs0 = gridspec.GridSpec(6, 2, figure=fig)
    ax1 = fig.add_subplot(gs0[:3, 1])
    ax2 = fig.add_subplot(gs0[3:, 1])
    example_plot(ax1, fontsize=18)
    example_plot(ax2, fontsize=18)
    ax = fig.add_subplot(gs0[0:2, 0])
    example_plot(ax, nodec=True)
    ax = fig.add_subplot(gs0[2:4, 0])
    example_plot(ax, nodec=True)
    ax = fig.add_subplot(gs0[4:, 0])
    example_plot(ax, nodec=True)
    ax.set_xlabel('x-label')