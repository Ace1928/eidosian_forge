import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
import matplotlib.transforms as mtransforms
@image_comparison(['streamplot_maxlength_no_broken.png'], remove_text=True, style='mpl20', tol=0.302)
def test_maxlength_no_broken():
    x, y, U, V = swirl_velocity_field()
    ax = plt.figure().subplots()
    ax.streamplot(x, y, U, V, maxlength=10.0, start_points=[[0.0, 1.5]], linewidth=2, density=2, broken_streamlines=False)
    assert ax.get_xlim()[-1] == ax.get_ylim()[-1] == 3
    ax.set(xlim=(None, 3.2555988021882305), ylim=(None, 3.078326760195413))