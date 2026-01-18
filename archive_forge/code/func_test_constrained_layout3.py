import gc
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import gridspec, ticker
@image_comparison(['constrained_layout3.png'])
def test_constrained_layout3():
    """Test constrained_layout for colorbars with subplots"""
    fig, axs = plt.subplots(2, 2, layout='constrained')
    for nn, ax in enumerate(axs.flat):
        pcm = example_pcolor(ax, fontsize=24)
        if nn == 3:
            pad = 0.08
        else:
            pad = 0.02
        fig.colorbar(pcm, ax=ax, pad=pad)