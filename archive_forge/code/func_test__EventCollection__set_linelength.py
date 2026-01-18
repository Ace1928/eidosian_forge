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
@image_comparison(['EventCollection_plot__set_linelength'])
def test__EventCollection__set_linelength():
    splt, coll, props = generate_EventCollection_plot()
    new_linelength = 15
    coll.set_linelength(new_linelength)
    assert new_linelength == coll.get_linelength()
    check_segments(coll, props['positions'], new_linelength, props['lineoffset'], props['orientation'])
    splt.set_title('EventCollection: set_linelength')
    splt.set_ylim(-20, 20)