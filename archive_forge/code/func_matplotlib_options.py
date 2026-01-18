import os
import sys
from IPython.external.qt_loaders import (
def matplotlib_options(mpl):
    """Constraints placed on an imported matplotlib."""
    if mpl is None:
        return
    backend = mpl.rcParams.get('backend', None)
    if backend == 'Qt4Agg':
        mpqt = mpl.rcParams.get('backend.qt4', None)
        if mpqt is None:
            return None
        if mpqt.lower() == 'pyside':
            return [QT_API_PYSIDE]
        elif mpqt.lower() == 'pyqt4':
            return [QT_API_PYQT_DEFAULT]
        elif mpqt.lower() == 'pyqt4v2':
            return [QT_API_PYQT]
        raise ImportError('unhandled value for backend.qt4 from matplotlib: %r' % mpqt)
    elif backend == 'Qt5Agg':
        mpqt = mpl.rcParams.get('backend.qt5', None)
        if mpqt is None:
            return None
        if mpqt.lower() == 'pyqt5':
            return [QT_API_PYQT5]
        raise ImportError('unhandled value for backend.qt5 from matplotlib: %r' % mpqt)