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
def test_array_dimensions(pcfunc):
    z = np.arange(12).reshape(3, 4)
    pc = getattr(plt, pcfunc)(z)
    pc.set_array(z.ravel())
    pc.update_scalarmappable()
    pc.set_array(z)
    pc.update_scalarmappable()
    z = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    pc.set_array(z)
    pc.update_scalarmappable()