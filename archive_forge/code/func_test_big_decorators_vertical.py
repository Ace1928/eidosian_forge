import warnings
import numpy as np
from numpy.testing import assert_array_equal
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea
from matplotlib.patches import Rectangle
def test_big_decorators_vertical():
    """Test that doesn't warn when ylabel too big."""
    fig, axs = plt.subplots(2, 1, figsize=(3, 2))
    axs[0].set_ylabel('a' * 20)
    axs[1].set_ylabel('b' * 20)