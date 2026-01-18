import sys, re, traceback, threading
from .. import exceptionHandling as exceptionHandling
from ..Qt import QtWidgets, QtCore
from ..functions import SignalBlock
from .stackwidget import StackWidget
def systrace(self, frame, event, arg):
    if event != 'exception':
        return self.systrace
    if self._inSystrace:
        return self.systrace
    self._inSystrace = True
    try:
        if self.checkException(*arg):
            self.exceptionHandler(arg[1], lastFrame=frame)
    except Exception as exc:
        print('Exception in systrace:')
        traceback.print_exc()
    finally:
        self.inSystrace = False
    return self.systrace