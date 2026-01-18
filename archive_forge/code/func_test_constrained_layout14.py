import gc
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import gridspec, ticker
@image_comparison(['constrained_layout14.png'])
def test_constrained_layout14():
    """Test that padding works."""
    fig, axs = plt.subplots(2, 2, layout='constrained')
    for ax in axs.flat:
        pcm = example_pcolor(ax, fontsize=12)
        fig.colorbar(pcm, ax=ax, shrink=0.6, aspect=20.0, pad=0.02)
    fig.get_layout_engine().set(w_pad=3.0 / 72.0, h_pad=3.0 / 72.0, hspace=0.2, wspace=0.2)