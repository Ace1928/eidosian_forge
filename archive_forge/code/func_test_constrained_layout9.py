import gc
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import gridspec, ticker
@image_comparison(['constrained_layout9.png'])
def test_constrained_layout9():
    """Test for handling suptitle and for sharex and sharey"""
    fig, axs = plt.subplots(2, 2, layout='constrained', sharex=False, sharey=False)
    for ax in axs.flat:
        pcm = example_pcolor(ax, fontsize=24)
        ax.set_xlabel('')
        ax.set_ylabel('')
    ax.set_aspect(2.0)
    fig.colorbar(pcm, ax=axs, pad=0.01, shrink=0.6)
    fig.suptitle('Test Suptitle', fontsize=28)