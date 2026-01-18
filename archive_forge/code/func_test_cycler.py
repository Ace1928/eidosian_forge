from io import BytesIO
import ast
import pickle
import pickletools
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib import cm
from matplotlib.testing import subprocess_run_helper
from matplotlib.testing.decorators import check_figures_equal
from matplotlib.dates import rrulewrapper
from matplotlib.lines import VertexSelector
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.figure as mfigure
from mpl_toolkits.axes_grid1 import parasite_axes  # type: ignore
def test_cycler():
    ax = plt.figure().add_subplot()
    ax.set_prop_cycle(c=['c', 'm', 'y', 'k'])
    ax.plot([1, 2])
    ax = pickle.loads(pickle.dumps(ax))
    l, = ax.plot([3, 4])
    assert l.get_color() == 'm'