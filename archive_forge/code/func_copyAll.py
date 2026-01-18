import numpy as np
from ..Qt import QtCore, QtGui, QtWidgets
def copyAll(self):
    """Copy all data to clipboard."""
    QtWidgets.QApplication.clipboard().setText(self.serialize(useSelection=False))