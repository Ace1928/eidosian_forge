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
def test_minor_accountedfor():
    with rc_context({'_internal.classic_mode': False}):
        fig, ax = plt.subplots(dpi=200, figsize=(6, 6))
        fig.canvas.draw()
        ax.tick_params(which='both', direction='out')
        bbaxis, bbspines, bbax, bbtb = color_boxes(fig, ax)
        bbaxis, bbspines, bbax, bbtb = color_boxes(fig, ax)
        targets = [[150.0, 108.88888888888886, 930.0, 11.111111111111114], [138.8889, 119.9999, 11.1111, 960.0]]
        for n in range(2):
            targetbb = mtransforms.Bbox.from_bounds(*targets[n])
            assert_allclose(bbspines[n * 2].bounds, targetbb.bounds, atol=0.01)
        fig, ax = plt.subplots(dpi=200, figsize=(6, 6))
        fig.canvas.draw()
        ax.tick_params(which='both', direction='out')
        ax.minorticks_on()
        ax.tick_params(axis='both', which='minor', length=30)
        fig.canvas.draw()
        bbaxis, bbspines, bbax, bbtb = color_boxes(fig, ax)
        targets = [[150.0, 36.66666666666663, 930.0, 83.33333333333334], [66.6667, 120.0, 83.3333, 960.0]]
        for n in range(2):
            targetbb = mtransforms.Bbox.from_bounds(*targets[n])
            assert_allclose(bbspines[n * 2].bounds, targetbb.bounds, atol=0.01)