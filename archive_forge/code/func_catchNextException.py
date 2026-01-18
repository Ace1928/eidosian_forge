import sys, re, traceback, threading
from .. import exceptionHandling as exceptionHandling
from ..Qt import QtWidgets, QtCore
from ..functions import SignalBlock
from .stackwidget import StackWidget
def catchNextException(self, catch=True):
    """
        If True, the console will catch the next unhandled exception and display the stack
        trace.
        """
    with SignalBlock(self.catchNextExceptionBtn.toggled, self.catchNextException):
        self.catchNextExceptionBtn.setChecked(catch)
    if catch:
        with SignalBlock(self.catchAllExceptionsBtn.toggled, self.catchAllExceptions):
            self.catchAllExceptionsBtn.setChecked(False)
        self.enableExceptionHandling()
    else:
        self.disableExceptionHandling()