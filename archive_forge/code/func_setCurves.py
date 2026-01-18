from .. import functions as fn
from ..Qt import QtGui, QtWidgets
from .PlotCurveItem import PlotCurveItem
from .PlotDataItem import PlotDataItem
def setCurves(self, curve1, curve2):
    """Set the curves to fill between.
        
        Arguments must be instances of PlotDataItem or PlotCurveItem.
        
        Added in version 0.9.9
        """
    if self.curves is not None:
        for c in self.curves:
            try:
                c.sigPlotChanged.disconnect(self.curveChanged)
            except (TypeError, RuntimeError):
                pass
    curves = [curve1, curve2]
    for c in curves:
        if not isinstance(c, PlotDataItem) and (not isinstance(c, PlotCurveItem)):
            raise TypeError('Curves must be PlotDataItem or PlotCurveItem.')
    self.curves = curves
    curve1.sigPlotChanged.connect(self.curveChanged)
    curve2.sigPlotChanged.connect(self.curveChanged)
    self.setZValue(min(curve1.zValue(), curve2.zValue()) - 1)
    self.curveChanged()