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
def test_cleanup_closepoly():
    paths = [Path([[np.nan, np.nan], [np.nan, np.nan]], [Path.MOVETO, Path.CLOSEPOLY]), Path([[np.nan, np.nan], [np.nan, np.nan]]), Path([[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]], [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.CLOSEPOLY])]
    for p in paths:
        cleaned = p.cleaned(remove_nans=True)
        assert len(cleaned) == 1
        assert cleaned.codes[0] == Path.STOP