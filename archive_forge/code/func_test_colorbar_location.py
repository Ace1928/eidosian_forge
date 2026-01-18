import gc
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import gridspec, ticker
@image_comparison(['test_colorbar_location.png'], remove_text=True, style='mpl20')
def test_colorbar_location():
    """
    Test that colorbar handling is as expected for various complicated
    cases...
    """
    plt.rcParams['pcolormesh.snap'] = False
    fig, axs = plt.subplots(4, 5, layout='constrained')
    for ax in axs.flat:
        pcm = example_pcolor(ax)
        ax.set_xlabel('')
        ax.set_ylabel('')
    fig.colorbar(pcm, ax=axs[:, 1], shrink=0.4)
    fig.colorbar(pcm, ax=axs[-1, :2], shrink=0.5, location='bottom')
    fig.colorbar(pcm, ax=axs[0, 2:], shrink=0.5, location='bottom', pad=0.05)
    fig.colorbar(pcm, ax=axs[-2, 3:], shrink=0.5, location='top')
    fig.colorbar(pcm, ax=axs[0, 0], shrink=0.5, location='left')
    fig.colorbar(pcm, ax=axs[1:3, 2], shrink=0.5, location='right')