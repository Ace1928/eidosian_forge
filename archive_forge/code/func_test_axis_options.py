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
@image_comparison(['axis_options.png'], remove_text=True, style='mpl20')
def test_axis_options():
    fig, axes = plt.subplots(2, 3)
    for i, option in enumerate(('scaled', 'tight', 'image')):
        axes[0, i].plot((1, 2), (1, 3.2))
        axes[0, i].axis(option)
        axes[0, i].add_artist(mpatches.Circle((1.5, 1.5), radius=0.5, facecolor='none', edgecolor='k'))
        axes[1, i].plot((1, 2.25), (1, 1.75))
        axes[1, i].axis(option)
        axes[1, i].add_artist(mpatches.Circle((1.5, 1.25), radius=0.25, facecolor='none', edgecolor='k'))