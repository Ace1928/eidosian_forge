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
def test_auto_no_rasterize():

    class Gen1(martist.Artist):
        ...
    assert 'draw' in Gen1.__dict__
    assert Gen1.__dict__['draw'] is Gen1.draw

    class Gen2(Gen1):
        ...
    assert 'draw' not in Gen2.__dict__
    assert Gen2.draw is Gen1.draw