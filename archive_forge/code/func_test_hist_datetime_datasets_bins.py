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
@pytest.mark.parametrize('bins_preprocess', [mpl.dates.date2num, lambda bins: bins, lambda bins: np.asarray(bins, 'datetime64')], ids=['date2num', 'datetime.datetime', 'np.datetime64'])
def test_hist_datetime_datasets_bins(bins_preprocess):
    data = [[datetime.datetime(2019, 1, 5), datetime.datetime(2019, 1, 11), datetime.datetime(2019, 2, 1), datetime.datetime(2019, 3, 1)], [datetime.datetime(2019, 1, 11), datetime.datetime(2019, 2, 5), datetime.datetime(2019, 2, 18), datetime.datetime(2019, 3, 1)]]
    date_edges = [datetime.datetime(2019, 1, 1), datetime.datetime(2019, 2, 1), datetime.datetime(2019, 3, 1)]
    fig, ax = plt.subplots()
    _, bins, _ = ax.hist(data, bins=bins_preprocess(date_edges), stacked=True)
    np.testing.assert_allclose(bins, mpl.dates.date2num(date_edges))
    _, bins, _ = ax.hist(data, bins=bins_preprocess(date_edges), stacked=False)
    np.testing.assert_allclose(bins, mpl.dates.date2num(date_edges))