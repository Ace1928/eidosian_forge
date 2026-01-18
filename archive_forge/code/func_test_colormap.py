import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
import matplotlib.transforms as mtransforms
@image_comparison(['streamplot_colormap'], remove_text=True, style='mpl20', tol=0.022)
def test_colormap():
    X, Y, U, V = velocity_field()
    plt.streamplot(X, Y, U, V, color=U, density=0.6, linewidth=2, cmap=plt.cm.autumn)
    plt.colorbar()