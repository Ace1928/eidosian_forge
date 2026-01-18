import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
import matplotlib.transforms as mtransforms
@image_comparison(['streamplot_direction.png'], remove_text=True, style='mpl20', tol=0.073)
def test_direction():
    x, y, U, V = swirl_velocity_field()
    plt.streamplot(x, y, U, V, integration_direction='backward', maxlength=1.5, start_points=[[1.5, 0.0]], linewidth=2, density=2)