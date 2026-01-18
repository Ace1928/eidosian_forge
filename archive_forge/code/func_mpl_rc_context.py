from contextlib import contextmanager
from itertools import chain
import matplotlib as mpl
import numpy as np
import param
from matplotlib import (
from matplotlib.font_manager import font_scalings
from mpl_toolkits.mplot3d import Axes3D  # noqa (For 3D plots)
from ...core import (
from ...core.options import SkipRendering, Store
from ...core.util import int_to_alpha, int_to_roman, wrap_tuple_streams
from ..plot import (
from ..util import attach_streams, collate, displayable
from .util import compute_ratios, fix_aspect, get_old_rcparams
def mpl_rc_context(f):
    """
    Decorator for MPLPlot methods applying the matplotlib rc params
    in the plots fig_rcparams while when method is called.
    """

    def wrapper(self, *args, **kwargs):
        with _rc_context(self.fig_rcparams):
            return f(self, *args, **kwargs)
    return wrapper