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
def showAxes(self, selection, showValues=True, size=False):
    """ 
        Convenience method for quickly configuring axis settings.
        
        Parameters
        ----------
        selection: bool or tuple of bool 
            Determines which AxisItems will be displayed.
            If in tuple form, order is (left, top, right, bottom)
            A single boolean value will set all axes, 
            so that ``showAxes(True)`` configures the axes to draw a frame.
        showValues: bool or tuple of bool, optional
            Determines if values will be displayed for the ticks of each axis.
            True value shows values for left and bottom axis (default).
            False shows no values.
            If in tuple form, order is (left, top, right, bottom)
            None leaves settings unchanged.
            If not specified, left and bottom axes will be drawn with values.
        size: float or tuple of float, optional
            Reserves as fixed amount of space (width for vertical axis, height for horizontal axis)
            for each axis where tick values are enabled. If only a single float value is given, it
            will be applied for both width and height. If `None` is given instead of a float value,
            the axis reverts to automatic allocation of space.
            If in tuple form, order is (width, height)
        """
    if selection is True:
        selection = (True, True, True, True)
    elif selection is False:
        selection = (False, False, False, False)
    if showValues is True:
        showValues = (True, False, False, True)
    elif showValues is False:
        showValues = (False, False, False, False)
    elif showValues is None:
        showValues = (None, None, None, None)
    if size is not False and (not isinstance(size, collections.abc.Sized)):
        size = (size, size)
    all_axes = ('left', 'top', 'right', 'bottom')
    for show_axis, show_value, axis_key in zip(selection, showValues, all_axes):
        if show_axis is None:
            pass
        elif show_axis:
            self.showAxis(axis_key)
        else:
            self.hideAxis(axis_key)
        if show_value is None:
            pass
        else:
            ax = self.getAxis(axis_key)
            ax.setStyle(showValues=show_value)
            if size is not False:
                if axis_key in ('left', 'right'):
                    if show_value:
                        ax.setWidth(size[0])
                    else:
                        ax.setWidth(None)
                elif axis_key in ('top', 'bottom'):
                    if show_value:
                        ax.setHeight(size[1])
                    else:
                        ax.setHeight(None)