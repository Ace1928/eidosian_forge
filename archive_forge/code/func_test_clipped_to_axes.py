import warnings
import numpy as np
from numpy.testing import assert_array_equal
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea
from matplotlib.patches import Rectangle
def test_clipped_to_axes():
    arr = np.arange(100).reshape((10, 10))
    fig = plt.figure(figsize=(6, 2))
    ax1 = fig.add_subplot(131, projection='rectilinear')
    ax2 = fig.add_subplot(132, projection='mollweide')
    ax3 = fig.add_subplot(133, projection='polar')
    for ax in (ax1, ax2, ax3):
        ax.grid(False)
        h, = ax.plot(arr[:, 0])
        m = ax.pcolor(arr)
        assert h._fully_clipped_to_axes()
        assert m._fully_clipped_to_axes()
        rect = Rectangle((0, 0), 0.5, 0.5, transform=ax.transAxes)
        h.set_clip_path(rect)
        m.set_clip_path(rect.get_path(), rect.get_transform())
        assert not h._fully_clipped_to_axes()
        assert not m._fully_clipped_to_axes()