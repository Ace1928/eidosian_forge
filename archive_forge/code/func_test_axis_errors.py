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
@pytest.mark.parametrize('err, args, kwargs, match', ((TypeError, (1, 2), {}, 'axis\\(\\) takes from 0 to 1 positional arguments but 2 were given'), (ValueError, ('foo',), {}, "Unrecognized string 'foo' to axis; try 'on' or 'off'"), (TypeError, ([1, 2],), {}, 'The first argument to axis*'), (TypeError, tuple(), {'foo': None}, "axis\\(\\) got an unexpected keyword argument 'foo'")))
def test_axis_errors(err, args, kwargs, match):
    with pytest.raises(err, match=match):
        plt.axis(*args, **kwargs)