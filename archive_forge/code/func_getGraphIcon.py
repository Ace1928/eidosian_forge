import os.path as op
import warnings
from ..Qt import QtGui, QtWidgets
def getGraphIcon(name):
    """Return a `PyQtGraph` icon from the registry by `name`"""
    icon = _ICON_REGISTRY[name]
    if isinstance(icon, GraphIcon):
        icon = icon.qicon
        _ICON_REGISTRY[name] = icon
    return icon