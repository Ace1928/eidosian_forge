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
def setDownsampling(self, ds=None, auto=None, mode=None):
    """
        Changes the default downsampling mode for all :class:`~pyqtgraph.PlotDataItem` managed by this plot.
        
        =============== ====================================================================
        **Arguments:**
        ds              (int) Reduce visible plot samples by this factor, or

                        (bool) To enable/disable downsampling without changing the value.

        auto            (bool) If `True`, automatically pick ``ds`` based on visible range

        mode            'subsample': Downsample by taking the first of N samples. This 
                        method is fastest but least accurate.

                        'mean': Downsample by taking the mean of N samples.

                        'peak': Downsample by drawing a saw wave that follows the min and 
                        max of the original data. This method produces the best visual 
                        representation of the data but is slower.
        =============== ====================================================================
        """
    if ds is not None:
        if ds is False:
            self.ctrl.downsampleCheck.setChecked(False)
        elif ds is True:
            self.ctrl.downsampleCheck.setChecked(True)
        else:
            self.ctrl.downsampleCheck.setChecked(True)
            self.ctrl.downsampleSpin.setValue(ds)
    if auto is not None:
        if auto and ds is not False:
            self.ctrl.downsampleCheck.setChecked(True)
        self.ctrl.autoDownsampleCheck.setChecked(auto)
    if mode is not None:
        if mode == 'subsample':
            self.ctrl.subsampleRadio.setChecked(True)
        elif mode == 'mean':
            self.ctrl.meanRadio.setChecked(True)
        elif mode == 'peak':
            self.ctrl.peakRadio.setChecked(True)
        else:
            raise ValueError("mode argument must be 'subsample', 'mean', or 'peak'.")