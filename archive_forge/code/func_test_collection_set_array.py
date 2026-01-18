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
def test_collection_set_array():
    vals = [*range(10)]
    c = Collection()
    c.set_array(vals)
    with pytest.raises(TypeError, match='^Image data of dtype'):
        c.set_array('wrong_input')
    vals[5] = 45
    assert np.not_equal(vals, c.get_array()).any()