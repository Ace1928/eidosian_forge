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
@mpl.style.context('default')
def test_color_logic(pcfunc):
    pcfunc = getattr(plt, pcfunc)
    z = np.arange(12).reshape(3, 4)
    pc = pcfunc(z, edgecolors='red', facecolors='none')
    pc.update_scalarmappable()
    face_default = mcolors.to_rgba_array(pc._get_default_facecolor())
    mapped = pc.get_cmap()(pc.norm(z.ravel()))
    assert mcolors.same_color(pc.get_edgecolor(), 'red')
    pc = pcfunc(z)
    pc.set_facecolor('none')
    pc.set_edgecolor('red')
    pc.update_scalarmappable()
    assert mcolors.same_color(pc.get_facecolor(), 'none')
    assert mcolors.same_color(pc.get_edgecolor(), [[1, 0, 0, 1]])
    pc.set_alpha(0.5)
    pc.update_scalarmappable()
    assert mcolors.same_color(pc.get_edgecolor(), [[1, 0, 0, 0.5]])
    pc.set_alpha(None)
    pc.update_scalarmappable()
    assert mcolors.same_color(pc.get_edgecolor(), [[1, 0, 0, 1]])
    pc.set_edgecolor(None)
    pc.update_scalarmappable()
    assert np.array_equal(pc.get_edgecolor(), mapped)
    pc.set_facecolor(None)
    pc.update_scalarmappable()
    assert np.array_equal(pc.get_facecolor(), mapped)
    assert mcolors.same_color(pc.get_edgecolor(), 'none')
    pc.set_array(None)
    pc.update_scalarmappable()
    assert mcolors.same_color(pc.get_edgecolor(), 'none')
    assert mcolors.same_color(pc.get_facecolor(), face_default)
    pc.set_array(z)
    pc.update_scalarmappable()
    assert np.array_equal(pc.get_facecolor(), mapped)
    assert mcolors.same_color(pc.get_edgecolor(), 'none')
    pc = pcfunc(z, edgecolors=(1, 0, 0), facecolors=(0, 1, 0))
    pc.update_scalarmappable()
    assert np.array_equal(pc.get_facecolor(), mapped)
    assert mcolors.same_color(pc.get_edgecolor(), [[1, 0, 0, 1]])
    pc = pcfunc(z, edgecolors=(1, 0, 0), facecolors=np.ones((12, 3)))
    pc.update_scalarmappable()
    assert np.array_equal(pc.get_facecolor(), mapped)
    assert mcolors.same_color(pc.get_edgecolor(), [[1, 0, 0, 1]])
    pc.set_array(None)
    pc.update_scalarmappable()
    assert mcolors.same_color(pc.get_facecolor(), np.ones((12, 3)))
    assert mcolors.same_color(pc.get_edgecolor(), [[1, 0, 0, 1]])
    pc = pcfunc(z, edgecolors=(1, 0, 0), facecolors=np.ones((12, 4)))
    pc.update_scalarmappable()
    assert np.array_equal(pc.get_facecolor(), mapped)
    assert mcolors.same_color(pc.get_edgecolor(), [[1, 0, 0, 1]])
    pc.set_array(None)
    pc.update_scalarmappable()
    assert mcolors.same_color(pc.get_facecolor(), np.ones((12, 4)))
    assert mcolors.same_color(pc.get_edgecolor(), [[1, 0, 0, 1]])