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
@image_comparison(['boxplot_mod_artists_after_plotting.png'], remove_text=True, savefig_kwarg={'dpi': 40}, style='default')
def test_boxplot_mod_artist_after_plotting():
    x = [0.15, 0.11, 0.06, 0.06, 0.12, 0.56, -0.56]
    fig, ax = plt.subplots()
    bp = ax.boxplot(x, sym='o')
    for key in bp:
        for obj in bp[key]:
            obj.set_color('green')