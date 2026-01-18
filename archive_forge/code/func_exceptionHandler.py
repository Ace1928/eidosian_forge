import sys, re, traceback, threading
from .. import exceptionHandling as exceptionHandling
from ..Qt import QtWidgets, QtCore
from ..functions import SignalBlock
from .stackwidget import StackWidget
def exceptionHandler(self, excInfo, lastFrame=None):
    if isinstance(excInfo, Exception):
        exc = excInfo
    else:
        exc = excInfo.exc_value
    isGuiThread = QtCore.QThread.currentThread() == QtCore.QCoreApplication.instance().thread()
    if not isGuiThread:
        self._threadException.emit((excInfo, lastFrame))
        return
    if self.catchNextExceptionBtn.isChecked():
        self.catchNextExceptionBtn.setChecked(False)
    elif not self.catchAllExceptionsBtn.isChecked():
        return
    self.setException(exc, lastFrame=lastFrame)