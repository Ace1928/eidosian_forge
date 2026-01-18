import sys
from functools import partial
from pydev_ipython.version import check_version
def import_pyqt4(version=2):
    """
    Import PyQt4

    Parameters
    ----------
    version : 1, 2, or None
      Which QString/QVariant API to use. Set to None to use the system
      default

    ImportErrors raised within this function are non-recoverable
    """
    import sip
    if version is not None:
        sip.setapi('QString', version)
        sip.setapi('QVariant', version)
    from PyQt4 import QtGui, QtCore, QtSvg
    if not check_version(QtCore.PYQT_VERSION_STR, '4.7'):
        raise ImportError('IPython requires PyQt4 >= 4.7, found %s' % QtCore.PYQT_VERSION_STR)
    QtCore.Signal = QtCore.pyqtSignal
    QtCore.Slot = QtCore.pyqtSlot
    version = sip.getapi('QString')
    api = QT_API_PYQTv1 if version == 1 else QT_API_PYQT
    return (QtCore, QtGui, QtSvg, api)