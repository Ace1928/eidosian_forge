import sys, re, traceback, threading
from .. import exceptionHandling as exceptionHandling
from ..Qt import QtWidgets, QtCore
from ..functions import SignalBlock
from .stackwidget import StackWidget
def catchAllExceptions(self, catch=True):
    """
        If True, the console will catch all unhandled exceptions and display the stack
        trace. Each exception caught clears the last.
        """
    with SignalBlock(self.catchAllExceptionsBtn.toggled, self.catchAllExceptions):
        self.catchAllExceptionsBtn.setChecked(catch)
    if catch:
        with SignalBlock(self.catchNextExceptionBtn.toggled, self.catchNextException):
            self.catchNextExceptionBtn.setChecked(False)
        self.enableExceptionHandling()
    else:
        self.disableExceptionHandling()