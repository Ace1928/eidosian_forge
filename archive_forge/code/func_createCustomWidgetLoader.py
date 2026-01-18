import logging
from pyside2uic.Compiler.indenter import write_code
from pyside2uic.Compiler.qtproxies import (QtWidgets, QtGui, Literal,
def createCustomWidgetLoader(self):
    cw = _CustomWidgetLoader()
    self._modules.append(cw)
    return cw