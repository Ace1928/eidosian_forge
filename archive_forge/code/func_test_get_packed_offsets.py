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
@pytest.mark.parametrize('widths', ([150], [150, 150, 150], [0.1], [0.1, 0.1]))
@pytest.mark.parametrize('total', (250, 100, 0, -1, None))
@pytest.mark.parametrize('sep', (250, 1, 0, -1))
@pytest.mark.parametrize('mode', ('expand', 'fixed', 'equal'))
def test_get_packed_offsets(widths, total, sep, mode):
    _get_packed_offsets(widths, total, sep, mode=mode)