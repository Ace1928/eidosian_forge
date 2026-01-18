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
def generate_EventCollection_plot():
    """Generate the initial collection and plot it."""
    positions = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0])
    extra_positions = np.array([34.0, 55.0, 89.0])
    orientation = 'horizontal'
    lineoffset = 1
    linelength = 0.5
    linewidth = 2
    color = [1, 0, 0, 1]
    linestyle = 'solid'
    antialiased = True
    coll = EventCollection(positions, orientation=orientation, lineoffset=lineoffset, linelength=linelength, linewidth=linewidth, color=color, linestyle=linestyle, antialiased=antialiased)
    fig, ax = plt.subplots()
    ax.add_collection(coll)
    ax.set_title('EventCollection: default')
    props = {'positions': positions, 'extra_positions': extra_positions, 'orientation': orientation, 'lineoffset': lineoffset, 'linelength': linelength, 'linewidth': linewidth, 'color': color, 'linestyle': linestyle, 'antialiased': antialiased}
    ax.set_xlim(-1, 22)
    ax.set_ylim(0, 2)
    return (ax, coll, props)