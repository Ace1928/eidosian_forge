import warnings
import numpy as np
from numpy.testing import assert_array_equal
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea
from matplotlib.patches import Rectangle
def test_manual_colorbar():
    fig, axes = plt.subplots(1, 2)
    pts = axes[1].scatter([0, 1], [0, 1], c=[1, 5])
    ax_rect = axes[1].get_position()
    cax = fig.add_axes([ax_rect.x1 + 0.005, ax_rect.y0, 0.015, ax_rect.height])
    fig.colorbar(pts, cax=cax)
    with pytest.warns(UserWarning, match='This figure includes Axes'):
        fig.tight_layout()