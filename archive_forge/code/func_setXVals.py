from .. import functions as fn
from ..Qt import QtGui, QtWidgets
from .UIGraphicsItem import UIGraphicsItem
def setXVals(self, vals):
    """Set the x values for the ticks. 
        
        ==============   =====================================================================
        **Arguments:**
        vals             A list of x values (in data/plot coordinates) at which to draw ticks.
        ==============   =====================================================================
        """
    self.xvals = vals
    self.rebuildTicks()