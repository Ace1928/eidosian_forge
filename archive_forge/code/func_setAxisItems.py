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
def setAxisItems(self, axisItems=None):
    """
        Place axis items as given by `axisItems`. Initializes non-existing axis items.
        
        ==============  ==========================================================================================
        **Arguments:**
        *axisItems*     Optional dictionary instructing the PlotItem to use pre-constructed items
                        for its axes. The dict keys must be axis names ('left', 'bottom', 'right', 'top')
                        and the values must be instances of AxisItem (or at least compatible with AxisItem).
        ==============  ==========================================================================================
        """
    if axisItems is None:
        axisItems = {}
    visibleAxes = ['left', 'bottom']
    visibleAxes.extend(axisItems.keys())
    for k, pos in (('top', (1, 1)), ('bottom', (3, 1)), ('left', (2, 0)), ('right', (2, 2))):
        if k in self.axes:
            if k not in axisItems:
                continue
            oldAxis = self.axes[k]['item']
            self.layout.removeItem(oldAxis)
            oldAxis.scene().removeItem(oldAxis)
            oldAxis.unlinkFromView()
        if k in axisItems:
            axis = axisItems[k]
            if axis.scene() is not None:
                if k not in self.axes or axis != self.axes[k]['item']:
                    raise RuntimeError("Can't add an axis to multiple plots. Shared axes can be achieved with multiple AxisItem instances and set[X/Y]Link.")
        else:
            axis = AxisItem(orientation=k, parent=self)
        axis.linkToView(self.vb)
        self.axes[k] = {'item': axis, 'pos': pos}
        self.layout.addItem(axis, *pos)
        axis.setZValue(0.5)
        axis.setFlag(axis.GraphicsItemFlag.ItemNegativeZStacksBehindParent)
        axisVisible = k in visibleAxes
        self.showAxis(k, axisVisible)