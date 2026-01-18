import gc
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import gridspec, ticker
@image_comparison(['constrained_layout6.png'], tol=0.002)
def test_constrained_layout6():
    """Test constrained_layout for nested gridspecs"""
    plt.rcParams['pcolormesh.snap'] = False
    fig = plt.figure(layout='constrained')
    gs = fig.add_gridspec(1, 2, figure=fig)
    gsl = gs[0].subgridspec(2, 2)
    gsr = gs[1].subgridspec(1, 2)
    axsl = []
    for gs in gsl:
        ax = fig.add_subplot(gs)
        axsl += [ax]
        example_plot(ax, fontsize=12)
    ax.set_xlabel('x-label\nMultiLine')
    axsr = []
    for gs in gsr:
        ax = fig.add_subplot(gs)
        axsr += [ax]
        pcm = example_pcolor(ax, fontsize=12)
    fig.colorbar(pcm, ax=axsr, pad=0.01, shrink=0.99, location='bottom', ticks=ticker.MaxNLocator(nbins=5))