import warnings
from ..Qt import QtCore, QtGui, QtWidgets
from ..widgets.VerticalLabel import VerticalLabel
from .DockDrop import DockDrop
def hideTitleBar(self):
    """
        Hide the title bar for this Dock.
        This will prevent the Dock being moved by the user.
        """
    self.label.hide()
    self.labelHidden = True
    self.dockdrop.removeAllowedArea('center')
    self.updateStyle()