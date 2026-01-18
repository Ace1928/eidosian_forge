from collections import namedtuple
import io
import numpy as np
from numpy.testing import assert_allclose
import pytest
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.offsetbox import (
def test_get_packed_offsets_equal_total_none_sep_none():
    with pytest.raises(ValueError):
        _get_packed_offsets([1, 1, 1], total=None, sep=None, mode='equal')