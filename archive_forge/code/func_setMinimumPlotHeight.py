from ..graphicsItems import MultiPlotItem as MultiPlotItem
from ..Qt import QtCore
from .GraphicsView import GraphicsView
def setMinimumPlotHeight(self, min):
    """Set the minimum height for each sub-plot displayed. 
        
        If the total height of all plots is greater than the height of the 
        widget, then a scroll bar will appear to provide access to the entire
        set of plots.
        
        Added in version 0.9.9
        """
    self.minPlotHeight = min
    self.resizeEvent(None)