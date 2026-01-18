import re
import numpy as np
from numpy.testing import assert_array_equal
import pytest
from matplotlib import patches
from matplotlib.path import Path
from matplotlib.patches import Polygon
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.backend_bases import MouseEvent
@image_comparison(['marker_paths.pdf'], remove_text=True)
def test_marker_paths_pdf():
    N = 7
    plt.errorbar(np.arange(N), np.ones(N) + 4, np.ones(N))
    plt.xlim(-1, N)
    plt.ylim(-1, 7)