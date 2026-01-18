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
@pytest.mark.parametrize('widths, total, sep, expected', [_Params([0.1, 0.1, 0.1], total=None, sep=None, expected=(1, [0, 0.45, 0.9])), _Params([3, 1, 2], total=10, sep=1, expected=(10, [0, 5, 8])), _Params([3, 1, 2], total=5, sep=1, expected=(5, [0, 2.5, 3]))])
def test_get_packed_offsets_expand(widths, total, sep, expected):
    result = _get_packed_offsets(widths, total, sep, mode='expand')
    assert result[0] == expected[0]
    assert_allclose(result[1], expected[1])