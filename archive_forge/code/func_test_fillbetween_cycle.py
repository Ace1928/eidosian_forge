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
@mpl.style.context('default')
def test_fillbetween_cycle():
    fig, ax = plt.subplots()
    for j in range(3):
        cc = ax.fill_between(range(3), range(3))
        target = mcolors.to_rgba(f'C{j}')
        assert tuple(cc.get_facecolors().squeeze()) == tuple(target)
    for j in range(3, 6):
        cc = ax.fill_betweenx(range(3), range(3))
        target = mcolors.to_rgba(f'C{j}')
        assert tuple(cc.get_facecolors().squeeze()) == tuple(target)
    target = mcolors.to_rgba('k')
    for al in ['facecolor', 'facecolors', 'color']:
        cc = ax.fill_between(range(3), range(3), **{al: 'k'})
        assert tuple(cc.get_facecolors().squeeze()) == tuple(target)
    edge_target = mcolors.to_rgba('k')
    for j, el in enumerate(['edgecolor', 'edgecolors'], start=6):
        cc = ax.fill_between(range(3), range(3), **{el: 'k'})
        face_target = mcolors.to_rgba(f'C{j}')
        assert tuple(cc.get_facecolors().squeeze()) == tuple(face_target)
        assert tuple(cc.get_edgecolors().squeeze()) == tuple(edge_target)