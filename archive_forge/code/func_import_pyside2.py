import sys
from functools import partial
from pydev_ipython.version import check_version
def import_pyside2():
    """
    Import PySide2

    ImportErrors raised within this function are non-recoverable
    """
    from PySide2 import QtGui, QtCore, QtSvg
    return (QtCore, QtGui, QtSvg, QT_API_PYSIDE)