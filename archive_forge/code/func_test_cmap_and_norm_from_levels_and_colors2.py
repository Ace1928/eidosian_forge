import copy
import itertools
import unittest.mock
from packaging.version import parse as parse_version
from io import BytesIO
import numpy as np
from PIL import Image
import pytest
import base64
from numpy.testing import assert_array_equal, assert_array_almost_equal
from matplotlib import cbook, cm
import matplotlib
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
import matplotlib.pyplot as plt
import matplotlib.scale as mscale
from matplotlib.rcsetup import cycler
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_cmap_and_norm_from_levels_and_colors2():
    levels = [-1, 2, 2.5, 3]
    colors = ['red', (0, 1, 0), 'blue', (0.5, 0.5, 0.5), (0.0, 0.0, 0.0, 1.0)]
    clr = mcolors.to_rgba_array(colors)
    bad = (0.1, 0.1, 0.1, 0.1)
    no_color = (0.0, 0.0, 0.0, 0.0)
    masked_value = 'masked_value'
    tests = [('both', None, {-2: clr[0], -1: clr[1], 2: clr[2], 2.25: clr[2], 3: clr[4], 3.5: clr[4], masked_value: bad}), ('min', -1, {-2: clr[0], -1: clr[1], 2: clr[2], 2.25: clr[2], 3: no_color, 3.5: no_color, masked_value: bad}), ('max', -1, {-2: no_color, -1: clr[0], 2: clr[1], 2.25: clr[1], 3: clr[3], 3.5: clr[3], masked_value: bad}), ('neither', -2, {-2: no_color, -1: clr[0], 2: clr[1], 2.25: clr[1], 3: no_color, 3.5: no_color, masked_value: bad})]
    for extend, i1, cases in tests:
        cmap, norm = mcolors.from_levels_and_colors(levels, colors[0:i1], extend=extend)
        cmap.set_bad(bad)
        for d_val, expected_color in cases.items():
            if d_val == masked_value:
                d_val = np.ma.array([1], mask=True)
            else:
                d_val = [d_val]
            assert_array_equal(expected_color, cmap(norm(d_val))[0], f'With extend={extend!r} and data value={d_val!r}')
    with pytest.raises(ValueError):
        mcolors.from_levels_and_colors(levels, colors)