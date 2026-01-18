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
@check_figures_equal()
def test_label_loc_rc(fig_test, fig_ref):
    with matplotlib.rc_context({'xaxis.labellocation': 'right', 'yaxis.labellocation': 'top'}):
        ax = fig_test.subplots()
        sc = ax.scatter([1, 2], [1, 2], c=[1, 2], label='scatter')
        ax.legend()
        ax.set_ylabel('Y Label')
        ax.set_xlabel('X Label')
        cbar = fig_test.colorbar(sc, orientation='horizontal')
        cbar.set_label('Z Label')
    ax = fig_ref.subplots()
    sc = ax.scatter([1, 2], [1, 2], c=[1, 2], label='scatter')
    ax.legend()
    ax.set_ylabel('Y Label', y=1, ha='right')
    ax.set_xlabel('X Label', x=1, ha='right')
    cbar = fig_ref.colorbar(sc, orientation='horizontal')
    cbar.set_label('Z Label', x=1, ha='right')