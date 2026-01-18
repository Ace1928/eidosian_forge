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
def set_inverted(self, inverted):
    """
        Set whether this Axis is oriented in the "inverse" direction.

        The "normal" direction is increasing to the right for the x-axis and to
        the top for the y-axis; the "inverse" direction is increasing to the
        left for the x-axis and to the bottom for the y-axis.
        """
    a, b = self.get_view_interval()
    self._set_lim(*sorted((a, b), reverse=bool(inverted)), auto=None)