import sys
import select
def enable_qt(self, app=None):
    from pydev_ipython.qt_for_kernel import QT_API, QT_API_PYQT5
    if QT_API == QT_API_PYQT5:
        self.enable_qt5(app)
    else:
        self.enable_qt4(app)