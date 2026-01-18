import sys, re, traceback, threading
from .. import exceptionHandling as exceptionHandling
from ..Qt import QtWidgets, QtCore
from ..functions import SignalBlock
from .stackwidget import StackWidget
def disableExceptionHandling(self):
    exceptionHandling.unregisterCallback(self.exceptionHandler)
    self.updateSysTrace()