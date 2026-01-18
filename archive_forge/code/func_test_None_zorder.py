import io
from itertools import chain
import numpy as np
import pytest
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.path as mpath
import matplotlib.transforms as mtransforms
import matplotlib.collections as mcollections
import matplotlib.artist as martist
import matplotlib.backend_bases as mbackend_bases
import matplotlib as mpl
from matplotlib.testing.decorators import check_figures_equal, image_comparison
def test_None_zorder():
    fig, ax = plt.subplots()
    ln, = ax.plot(range(5), zorder=None)
    assert ln.get_zorder() == mlines.Line2D.zorder
    ln.set_zorder(123456)
    assert ln.get_zorder() == 123456
    ln.set_zorder(None)
    assert ln.get_zorder() == mlines.Line2D.zorder