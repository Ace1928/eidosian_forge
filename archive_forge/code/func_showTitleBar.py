import warnings
from ..Qt import QtCore, QtGui, QtWidgets
from ..widgets.VerticalLabel import VerticalLabel
from .DockDrop import DockDrop
def showTitleBar(self):
    """
        Show the title bar for this Dock.
        """
    self.label.show()
    self.labelHidden = False
    self.dockdrop.addAllowedArea('center')
    self.updateStyle()