import gc
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import gridspec, ticker
@image_comparison(['test_colorbars_no_overlapV.png'], style='mpl20')
def test_colorbars_no_overlapV():
    fig = plt.figure(figsize=(2, 4), layout='constrained')
    axs = fig.subplots(2, 1, sharex=True, sharey=True)
    for ax in axs:
        ax.yaxis.set_major_formatter(ticker.NullFormatter())
        ax.tick_params(axis='both', direction='in')
        im = ax.imshow([[1, 2], [3, 4]])
        fig.colorbar(im, ax=ax, orientation='vertical')
    fig.suptitle('foo')