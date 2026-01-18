import warnings
from ..Qt import QtCore, QtGui, QtWidgets
from ..widgets.VerticalLabel import VerticalLabel
from .DockDrop import DockDrop
def setStretch(self, x=None, y=None):
    """
        Set the 'target' size for this Dock.
        The actual size will be determined by comparing this Dock's
        stretch value to the rest of the docks it shares space with.
        """
    if x is None:
        x = 0
    if y is None:
        y = 0
    self._stretch = (x, y)
    self.sigStretchChanged.emit()