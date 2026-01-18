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
def test_bar_pandas(pd):
    df = pd.DataFrame({'year': [2018, 2018, 2018], 'month': [1, 1, 1], 'day': [1, 2, 3], 'value': [1, 2, 3]})
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    monthly = df[['date', 'value']].groupby(['date']).sum()
    dates = monthly.index
    forecast = monthly['value']
    baseline = monthly['value']
    fig, ax = plt.subplots()
    ax.bar(dates, forecast, width=10, align='center')
    ax.plot(dates, baseline, color='orange', lw=4)