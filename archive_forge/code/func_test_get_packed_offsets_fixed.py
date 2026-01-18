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
@pytest.mark.parametrize('widths, total, sep, expected', [_Params([3, 1, 2], total=None, sep=1, expected=(8, [0, 4, 6])), _Params([3, 1, 2], total=10, sep=1, expected=(10, [0, 4, 6])), _Params([3, 1, 2], total=5, sep=1, expected=(5, [0, 4, 6]))])
def test_get_packed_offsets_fixed(widths, total, sep, expected):
    result = _get_packed_offsets(widths, total, sep, mode='fixed')
    assert result[0] == expected[0]
    assert_allclose(result[1], expected[1])