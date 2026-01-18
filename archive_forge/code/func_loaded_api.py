import sys
from functools import partial
from pydev_ipython.version import check_version
def loaded_api():
    """Return which API is loaded, if any

    If this returns anything besides None,
    importing any other Qt binding is unsafe.

    Returns
    -------
    None, 'pyside', 'pyside2', 'pyqt', or 'pyqtv1'
    """
    if 'PyQt4.QtCore' in sys.modules:
        if qtapi_version() == 2:
            return QT_API_PYQT
        else:
            return QT_API_PYQTv1
    elif 'PySide.QtCore' in sys.modules:
        return QT_API_PYSIDE
    elif 'PySide2.QtCore' in sys.modules:
        return QT_API_PYSIDE2
    elif 'PyQt5.QtCore' in sys.modules:
        return QT_API_PYQT5
    return None