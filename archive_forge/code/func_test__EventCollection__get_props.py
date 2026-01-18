from datetime import datetime
import io
import itertools
import re
from types import SimpleNamespace
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.collections as mcollections
import matplotlib.colors as mcolors
import matplotlib.path as mpath
import matplotlib.transforms as mtransforms
from matplotlib.collections import (Collection, LineCollection,
from matplotlib.testing.decorators import check_figures_equal, image_comparison
@image_comparison(['EventCollection_plot__default'])
def test__EventCollection__get_props():
    _, coll, props = generate_EventCollection_plot()
    check_segments(coll, props['positions'], props['linelength'], props['lineoffset'], props['orientation'])
    np.testing.assert_array_equal(props['positions'], coll.get_positions())
    assert props['orientation'] == coll.get_orientation()
    assert coll.is_horizontal()
    assert props['linelength'] == coll.get_linelength()
    assert props['lineoffset'] == coll.get_lineoffset()
    assert coll.get_linestyle() == [(0, None)]
    for color in [coll.get_color(), *coll.get_colors()]:
        np.testing.assert_array_equal(color, props['color'])