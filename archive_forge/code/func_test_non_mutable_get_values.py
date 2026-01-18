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
@pytest.mark.parametrize('kind', ('over', 'under', 'bad'))
def test_non_mutable_get_values(kind):
    cmap = copy.copy(mpl.colormaps['viridis'])
    init_value = getattr(cmap, f'get_{kind}')()
    getattr(cmap, f'set_{kind}')('k')
    black_value = getattr(cmap, f'get_{kind}')()
    assert np.all(black_value == [0, 0, 0, 1])
    assert not np.all(init_value == black_value)