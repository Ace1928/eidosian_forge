import warnings
import numpy as np
from numpy.testing import assert_array_equal
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea
from matplotlib.patches import Rectangle
def test_big_decorators_horizontal():
    """Test that doesn't warn when xlabel too big."""
    fig, axs = plt.subplots(1, 2, figsize=(3, 2))
    axs[0].set_xlabel('a' * 30)
    axs[1].set_xlabel('b' * 30)