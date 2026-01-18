import logging
from pyside2uic.Compiler.indenter import write_code
from pyside2uic.Compiler.qtproxies import (QtWidgets, QtGui, Literal,
def createQtWidgetsWrapper(self):
    return _QtWidgetsWrapper