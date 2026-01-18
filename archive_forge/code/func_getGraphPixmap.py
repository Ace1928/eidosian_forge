import os.path as op
import warnings
from ..Qt import QtGui, QtWidgets
def getGraphPixmap(name, size=(20, 20)):
    """Return a `QPixmap` from the registry by `name`"""
    icon = getGraphIcon(name)
    return icon.pixmap(*size)