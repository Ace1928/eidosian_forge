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
@image_comparison(['EventCollection_plot__add_positions'])
def test__EventCollection__add_positions():
    splt, coll, props = generate_EventCollection_plot()
    new_positions = np.hstack([props['positions'], props['extra_positions'][0]])
    coll.switch_orientation()
    coll.add_positions(props['extra_positions'][0])
    coll.switch_orientation()
    np.testing.assert_array_equal(new_positions, coll.get_positions())
    check_segments(coll, new_positions, props['linelength'], props['lineoffset'], props['orientation'])
    splt.set_title('EventCollection: add_positions')
    splt.set_xlim(-1, 35)