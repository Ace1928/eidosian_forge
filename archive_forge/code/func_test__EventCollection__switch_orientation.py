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
@image_comparison(['EventCollection_plot__switch_orientation'])
def test__EventCollection__switch_orientation():
    splt, coll, props = generate_EventCollection_plot()
    new_orientation = 'vertical'
    coll.switch_orientation()
    assert new_orientation == coll.get_orientation()
    assert not coll.is_horizontal()
    new_positions = coll.get_positions()
    check_segments(coll, new_positions, props['linelength'], props['lineoffset'], new_orientation)
    splt.set_title('EventCollection: switch_orientation')
    splt.set_ylim(-1, 22)
    splt.set_xlim(0, 2)