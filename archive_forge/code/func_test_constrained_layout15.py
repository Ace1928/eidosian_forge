import gc
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import gridspec, ticker
@image_comparison(['constrained_layout15.png'])
def test_constrained_layout15():
    """Test that rcparams work."""
    mpl.rcParams['figure.constrained_layout.use'] = True
    fig, axs = plt.subplots(2, 2)
    for ax in axs.flat:
        example_plot(ax, fontsize=12)