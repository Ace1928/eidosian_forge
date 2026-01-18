import gc
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import gridspec, ticker
@image_comparison(['constrained_layout8.png'])
def test_constrained_layout8():
    """Test for gridspecs that are not completely full"""
    fig = plt.figure(figsize=(10, 5), layout='constrained')
    gs = gridspec.GridSpec(3, 5, figure=fig)
    axs = []
    for j in [0, 1]:
        if j == 0:
            ilist = [1]
        else:
            ilist = [0, 4]
        for i in ilist:
            ax = fig.add_subplot(gs[j, i])
            axs += [ax]
            example_pcolor(ax, fontsize=9)
            if i > 0:
                ax.set_ylabel('')
            if j < 1:
                ax.set_xlabel('')
            ax.set_title('')
    ax = fig.add_subplot(gs[2, :])
    axs += [ax]
    pcm = example_pcolor(ax, fontsize=9)
    fig.colorbar(pcm, ax=axs, pad=0.01, shrink=0.6)