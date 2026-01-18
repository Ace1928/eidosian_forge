import gc
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import gridspec, ticker
def test_hidden_axes():
    fig, axs = plt.subplots(2, 2, layout='constrained')
    axs[0, 1].set_visible(False)
    fig.draw_without_rendering()
    extents1 = np.copy(axs[0, 0].get_position().extents)
    np.testing.assert_allclose(extents1, [0.045552, 0.543288, 0.47819, 0.982638], rtol=1e-05)