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
@image_comparison(['arrow_simple.png'], remove_text=True)
def test_arrow_simple():
    length_includes_head = (True, False)
    shape = ('full', 'left', 'right')
    head_starts_at_zero = (True, False)
    kwargs = product(length_includes_head, shape, head_starts_at_zero)
    fig, axs = plt.subplots(3, 4)
    for i, (ax, kwarg) in enumerate(zip(axs.flat, kwargs)):
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        length_includes_head, shape, head_starts_at_zero = kwarg
        theta = 2 * np.pi * i / 12
        ax.arrow(0, 0, np.sin(theta), np.cos(theta), width=theta / 100, length_includes_head=length_includes_head, shape=shape, head_starts_at_zero=head_starts_at_zero, head_width=theta / 10, head_length=theta / 10)