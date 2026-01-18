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
@_api.deprecated('3.6', alternative='fig.get_layout_engine().get()', pending=True)
def get_constrained_layout_pads(self, relative=False):
    """
        Get padding for ``constrained_layout``.

        Returns a list of ``w_pad, h_pad`` in inches and
        ``wspace`` and ``hspace`` as fractions of the subplot.
        All values are None if ``constrained_layout`` is not used.

        See :ref:`constrainedlayout_guide`.

        Parameters
        ----------
        relative : bool
            If `True`, then convert from inches to figure relative.
        """
    if not isinstance(self.get_layout_engine(), ConstrainedLayoutEngine):
        return (None, None, None, None)
    info = self.get_layout_engine().get()
    w_pad = info['w_pad']
    h_pad = info['h_pad']
    wspace = info['wspace']
    hspace = info['hspace']
    if relative and (w_pad is not None or h_pad is not None):
        renderer = self._get_renderer()
        dpi = renderer.dpi
        w_pad = w_pad * dpi / renderer.width
        h_pad = h_pad * dpi / renderer.height
    return (w_pad, h_pad, wspace, hspace)