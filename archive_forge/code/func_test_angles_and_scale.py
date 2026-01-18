import platform
import sys
import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.testing.decorators import image_comparison
def test_angles_and_scale():
    fig, ax = plt.subplots()
    X, Y = np.meshgrid(np.arange(15), np.arange(10))
    U = V = np.ones_like(X)
    phi = (np.random.rand(15, 10) - 0.5) * 150
    ax.quiver(X, Y, U, V, angles=phi, scale_units='xy')