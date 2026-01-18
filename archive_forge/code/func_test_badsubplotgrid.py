import warnings
import numpy as np
from numpy.testing import assert_array_equal
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea
from matplotlib.patches import Rectangle
def test_badsubplotgrid():
    plt.subplot2grid((4, 5), (0, 0))
    plt.subplot2grid((5, 5), (0, 3), colspan=3, rowspan=5)
    with pytest.warns(UserWarning):
        plt.tight_layout()