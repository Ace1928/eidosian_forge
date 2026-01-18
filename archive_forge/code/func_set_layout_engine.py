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
def set_layout_engine(self, layout=None, **kwargs):
    """
        Set the layout engine for this figure.

        Parameters
        ----------
        layout : {'constrained', 'compressed', 'tight', 'none', `.LayoutEngine`, None}

            - 'constrained' will use `~.ConstrainedLayoutEngine`
            - 'compressed' will also use `~.ConstrainedLayoutEngine`, but with
              a correction that attempts to make a good layout for fixed-aspect
              ratio Axes.
            - 'tight' uses `~.TightLayoutEngine`
            - 'none' removes layout engine.

            If a `.LayoutEngine` instance, that instance will be used.

            If `None`, the behavior is controlled by :rc:`figure.autolayout`
            (which if `True` behaves as if 'tight' was passed) and
            :rc:`figure.constrained_layout.use` (which if `True` behaves as if
            'constrained' was passed).  If both are `True`,
            :rc:`figure.autolayout` takes priority.

            Users and libraries can define their own layout engines and pass
            the instance directly as well.

        **kwargs
            The keyword arguments are passed to the layout engine to set things
            like padding and margin sizes.  Only used if *layout* is a string.

        """
    if layout is None:
        if mpl.rcParams['figure.autolayout']:
            layout = 'tight'
        elif mpl.rcParams['figure.constrained_layout.use']:
            layout = 'constrained'
        else:
            self._layout_engine = None
            return
    if layout == 'tight':
        new_layout_engine = TightLayoutEngine(**kwargs)
    elif layout == 'constrained':
        new_layout_engine = ConstrainedLayoutEngine(**kwargs)
    elif layout == 'compressed':
        new_layout_engine = ConstrainedLayoutEngine(compress=True, **kwargs)
    elif layout == 'none':
        if self._layout_engine is not None:
            new_layout_engine = PlaceHolderLayoutEngine(self._layout_engine.adjust_compatible, self._layout_engine.colorbar_gridspec)
        else:
            new_layout_engine = None
    elif isinstance(layout, LayoutEngine):
        new_layout_engine = layout
    else:
        raise ValueError(f"Invalid value for 'layout': {layout!r}")
    if self._check_layout_engines_compat(self._layout_engine, new_layout_engine):
        self._layout_engine = new_layout_engine
    else:
        raise RuntimeError('Colorbar layout of new layout engine not compatible with old engine, and a colorbar has been created.  Engine not changed.')