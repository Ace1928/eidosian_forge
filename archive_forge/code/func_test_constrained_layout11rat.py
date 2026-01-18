import gc
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import gridspec, ticker
@image_comparison(['constrained_layout11rat.png'])
def test_constrained_layout11rat():
    """Test for multiple nested gridspecs with width_ratios"""
    fig = plt.figure(layout='constrained', figsize=(10, 3))
    gs0 = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[6, 1])
    gsl = gridspec.GridSpecFromSubplotSpec(1, 2, gs0[0])
    gsl0 = gridspec.GridSpecFromSubplotSpec(2, 2, gsl[1], height_ratios=[2, 1])
    ax = fig.add_subplot(gs0[1])
    example_plot(ax, fontsize=9)
    axs = []
    for gs in gsl0:
        ax = fig.add_subplot(gs)
        axs += [ax]
        pcm = example_pcolor(ax, fontsize=9)
    fig.colorbar(pcm, ax=axs, shrink=0.6, aspect=70.0)
    ax = fig.add_subplot(gsl[0])
    example_plot(ax, fontsize=9)