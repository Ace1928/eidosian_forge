from .. import debug
from .. import functions as fn
from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject
from .InfiniteLine import InfiniteLine
def setRegion(self, rgn):
    """Set the values for the edges of the region.
        
        ==============   ==============================================
        **Arguments:**
        rgn              A list or tuple of the lower and upper values.
        ==============   ==============================================
        """
    if self.lines[0].value() == rgn[0] and self.lines[1].value() == rgn[1]:
        return
    self.blockLineSignal = True
    self.lines[0].setValue(rgn[0])
    self.blockLineSignal = False
    self.lines[1].setValue(rgn[1])
    self.lineMoved(0)
    self.lineMoved(1)
    self.lineMoveFinished()