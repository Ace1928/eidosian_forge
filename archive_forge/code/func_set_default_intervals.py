import datetime
import functools
import logging
from numbers import Real
import warnings
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
import matplotlib.artist as martist
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.scale as mscale
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import matplotlib.units as munits
def set_default_intervals(self):
    if not self.axes.dataLim.mutatedy() and (not self.axes.viewLim.mutatedy()):
        if self.converter is not None:
            info = self.converter.axisinfo(self.units, self)
            if info.default_limits is not None:
                ymin, ymax = self.convert_units(info.default_limits)
                self.axes.viewLim.intervaly = (ymin, ymax)
    self.stale = True