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
def set_offset_position(self, position):
    """
        Parameters
        ----------
        position : {'left', 'right'}
        """
    x, y = self.offsetText.get_position()
    x = _api.check_getitem({'left': 0, 'right': 1}, position=position)
    self.offsetText.set_ha(position)
    self.offsetText.set_position((x, y))
    self.stale = True