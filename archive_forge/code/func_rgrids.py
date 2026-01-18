from __future__ import annotations
from contextlib import AbstractContextManager, ExitStack
from enum import Enum
import functools
import importlib
import inspect
import logging
import re
import sys
import threading
import time
from typing import cast, overload
from cycler import cycler
import matplotlib
import matplotlib.colorbar
import matplotlib.image
from matplotlib import _api
from matplotlib import (  # Re-exported for typing.
from matplotlib import _pylab_helpers, interactive
from matplotlib import cbook
from matplotlib import _docstring
from matplotlib.backend_bases import (
from matplotlib.figure import Figure, FigureBase, figaspect
from matplotlib.gridspec import GridSpec, SubplotSpec
from matplotlib import rcsetup, rcParamsDefault, rcParamsOrig
from matplotlib.artist import Artist
from matplotlib.axes import Axes, Subplot  # type: ignore
from matplotlib.projections import PolarAxes  # type: ignore
from matplotlib import mlab  # for detrend_none, window_hanning
from matplotlib.scale import get_scale_names
from matplotlib.cm import _colormaps
from matplotlib.cm import register_cmap  # type: ignore
from matplotlib.colors import _color_sequences
import numpy as np
from typing import TYPE_CHECKING, cast
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D, AxLine
from matplotlib.text import Text, Annotation
from matplotlib.patches import Polygon, Rectangle, Circle, Arrow
from matplotlib.widgets import Button, Slider, Widget
from .ticker import (
def rgrids(radii: ArrayLike | None=None, labels: Sequence[str | Text] | None=None, angle: float | None=None, fmt: str | None=None, **kwargs) -> tuple[list[Line2D], list[Text]]:
    """
    Get or set the radial gridlines on the current polar plot.

    Call signatures::

     lines, labels = rgrids()
     lines, labels = rgrids(radii, labels=None, angle=22.5, fmt=None, **kwargs)

    When called with no arguments, `.rgrids` simply returns the tuple
    (*lines*, *labels*). When called with arguments, the labels will
    appear at the specified radial distances and angle.

    Parameters
    ----------
    radii : tuple with floats
        The radii for the radial gridlines

    labels : tuple with strings or None
        The labels to use at each radial gridline. The
        `matplotlib.ticker.ScalarFormatter` will be used if None.

    angle : float
        The angular position of the radius labels in degrees.

    fmt : str or None
        Format string used in `matplotlib.ticker.FormatStrFormatter`.
        For example '%f'.

    Returns
    -------
    lines : list of `.lines.Line2D`
        The radial gridlines.

    labels : list of `.text.Text`
        The tick labels.

    Other Parameters
    ----------------
    **kwargs
        *kwargs* are optional `.Text` properties for the labels.

    See Also
    --------
    .pyplot.thetagrids
    .projections.polar.PolarAxes.set_rgrids
    .Axis.get_gridlines
    .Axis.get_ticklabels

    Examples
    --------
    ::

      # set the locations of the radial gridlines
      lines, labels = rgrids( (0.25, 0.5, 1.0) )

      # set the locations and labels of the radial gridlines
      lines, labels = rgrids( (0.25, 0.5, 1.0), ('Tom', 'Dick', 'Harry' ))
    """
    ax = gca()
    if not isinstance(ax, PolarAxes):
        raise RuntimeError('rgrids only defined for polar axes')
    if all((p is None for p in [radii, labels, angle, fmt])) and (not kwargs):
        lines_out: list[Line2D] = ax.yaxis.get_gridlines()
        labels_out: list[Text] = ax.yaxis.get_ticklabels()
    elif radii is None:
        raise TypeError("'radii' cannot be None when other parameters are passed")
    else:
        lines_out, labels_out = ax.set_rgrids(radii, labels=labels, angle=angle, fmt=fmt, **kwargs)
    return (lines_out, labels_out)