from operator import methodcaller
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
import matplotlib.artist as martist
import matplotlib.colors as mcolors
import matplotlib.text as mtext
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.transforms import (
from .axisline_style import AxislineStyle
def set_pad(self, pad):
    """
        Set the internal pad in points.

        The actual pad will be the sum of the internal pad and the
        external pad (the latter is set automatically by the `.AxisArtist`).

        Parameters
        ----------
        pad : float
            The internal pad in points.
        """
    self._pad = pad