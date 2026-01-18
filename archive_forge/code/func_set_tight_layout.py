from contextlib import ExitStack
import inspect
import itertools
import logging
from numbers import Integral
import threading
import numpy as np
import matplotlib as mpl
from matplotlib import _blocking_input, backend_bases, _docstring, projections
from matplotlib.artist import (
from matplotlib.backend_bases import (
import matplotlib._api as _api
import matplotlib.cbook as cbook
import matplotlib.colorbar as cbar
import matplotlib.image as mimage
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.layout_engine import (
import matplotlib.legend as mlegend
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.transforms import (Affine2D, Bbox, BboxTransformTo,
@_api.deprecated('3.6', alternative='set_layout_engine', pending=True)
def set_tight_layout(self, tight):
    """
        Set whether and how `.tight_layout` is called when drawing.

        Parameters
        ----------
        tight : bool or dict with keys "pad", "w_pad", "h_pad", "rect" or None
            If a bool, sets whether to call `.tight_layout` upon drawing.
            If ``None``, use :rc:`figure.autolayout` instead.
            If a dict, pass it as kwargs to `.tight_layout`, overriding the
            default paddings.
        """
    if tight is None:
        tight = mpl.rcParams['figure.autolayout']
    _tight = 'tight' if bool(tight) else 'none'
    _tight_parameters = tight if isinstance(tight, dict) else {}
    self.set_layout_engine(_tight, **_tight_parameters)
    self.stale = True