import collections.abc
import os
import warnings
import weakref
import numpy as np
from ... import functions as fn
from ... import icons
from ...Qt import QtCore, QtWidgets
from ...WidgetGroup import WidgetGroup
from ...widgets.FileDialog import FileDialog
from ..AxisItem import AxisItem
from ..ButtonItem import ButtonItem
from ..GraphicsWidget import GraphicsWidget
from ..InfiniteLine import InfiniteLine
from ..LabelItem import LabelItem
from ..LegendItem import LegendItem
from ..PlotCurveItem import PlotCurveItem
from ..PlotDataItem import PlotDataItem
from ..ScatterPlotItem import ScatterPlotItem
from ..ViewBox import ViewBox
from . import plotConfigTemplate_generic as ui_template
def multiDataPlot(self, *, x=None, y=None, constKwargs=None, **kwargs):
    """
        Allow plotting multiple curves on the same plot, changing some kwargs
        per curve.

        Parameters
        ----------
        x, y: array_like
            can be in the following formats:
              - {x or y} = [n1, n2, n3, ...]: The named argument iterates through
                ``n`` curves, while the unspecified argument is range(len(n)) for
                each curve.
              - x, [y1, y2, y3, ...]
              - [x1, x2, x3, ...], [y1, y2, y3, ...]
              - [x1, x2, x3, ...], y

              where ``x_n`` and ``y_n`` are ``ndarray`` data for each curve. Since
              ``x`` and ``y`` values are matched using ``zip``, unequal lengths mean
              the longer array will be truncated. Note that 2D matrices for either x
              or y are considered lists of curve
              data.
        constKwargs: dict, optional
            A dict of {str: value} passed to each curve during ``plot()``.
        kwargs: dict, optional
            A dict of {str: iterable} where the str is the name of a kwarg and the
            iterable is a list of values, one for each plotted curve.
        """
    if x is not None and (not len(x)) or (y is not None and (not len(y))):
        return []

    def scalarOrNone(val):
        return val is None or (len(val) and np.isscalar(val[0]))
    if scalarOrNone(x) and scalarOrNone(y):
        raise ValueError('If both `x` and `y` represent single curves, use `plot` instead of `multiPlot`.')
    curves = []
    constKwargs = constKwargs or {}
    xy: 'dict[str, list | None]' = dict(x=x, y=y)
    for key, oppositeVal in zip(('x', 'y'), [y, x]):
        oppositeVal: 'Iterable | None'
        val = xy[key]
        if val is None:
            val = range(max((len(curveN) for curveN in oppositeVal)))
        if np.isscalar(val[0]):
            val = [val] * len(oppositeVal)
        xy[key] = val
    for ii, (xi, yi) in enumerate(zip(xy['x'], xy['y'])):
        for kk in kwargs:
            if len(kwargs[kk]) <= ii:
                raise ValueError(f'Not enough values for kwarg `{kk}`. Expected {ii + 1:d} (number of curves to plot), got {len(kwargs[kk]):d}')
        kwargsi = {kk: kwargs[kk][ii] for kk in kwargs}
        constKwargs.update(kwargsi)
        curves.append(self.plot(xi, yi, **constKwargs))
    return curves