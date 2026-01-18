import contextlib
from collections import namedtuple
import datetime
from decimal import Decimal
from functools import partial
import inspect
import io
from itertools import product
import platform
from types import SimpleNamespace
import dateutil.tz
import numpy as np
from numpy import ma
from cycler import cycler
import pytest
import matplotlib
import matplotlib as mpl
from matplotlib import rc_context, patheffects
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.font_manager as mfont_manager
import matplotlib.markers as mmarkers
import matplotlib.patches as mpatches
import matplotlib.path as mpath
from matplotlib.projections.geo import HammerAxes
from matplotlib.projections.polar import PolarAxes
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import mpl_toolkits.axisartist as AA  # type: ignore
from numpy.testing import (
from matplotlib.testing.decorators import (
@image_comparison(['eventplot', 'eventplot'], remove_text=True)
def test_eventplot():
    np.random.seed(0)
    data1 = np.random.random([32, 20]).tolist()
    data2 = np.random.random([6, 20]).tolist()
    data = data1 + data2
    num_datasets = len(data)
    colors1 = [[0, 1, 0.7]] * len(data1)
    colors2 = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0.75, 0], [1, 0, 1], [0, 1, 1]]
    colors = colors1 + colors2
    lineoffsets1 = 12 + np.arange(0, len(data1)) * 0.33
    lineoffsets2 = [-15, -3, 1, 1.5, 6, 10]
    lineoffsets = lineoffsets1.tolist() + lineoffsets2
    linelengths1 = [0.33] * len(data1)
    linelengths2 = [5, 2, 1, 1, 3, 1.5]
    linelengths = linelengths1 + linelengths2
    fig = plt.figure()
    axobj = fig.add_subplot()
    colls = axobj.eventplot(data, colors=colors, lineoffsets=lineoffsets, linelengths=linelengths)
    num_collections = len(colls)
    assert num_collections == num_datasets
    data = {'pos': data, 'c': colors, 'lo': lineoffsets, 'll': linelengths}
    fig = plt.figure()
    axobj = fig.add_subplot()
    colls = axobj.eventplot('pos', colors='c', lineoffsets='lo', linelengths='ll', data=data)
    num_collections = len(colls)
    assert num_collections == num_datasets