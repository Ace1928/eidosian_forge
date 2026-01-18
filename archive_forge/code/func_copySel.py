import numpy as np
from ..Qt import QtCore, QtGui, QtWidgets
def copySel(self):
    """Copy selected data to clipboard."""
    QtWidgets.QApplication.clipboard().setText(self.serialize(useSelection=True))