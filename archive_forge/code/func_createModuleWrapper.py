import logging
from pyside2uic.Compiler.indenter import write_code
from pyside2uic.Compiler.qtproxies import (QtWidgets, QtGui, Literal,
def createModuleWrapper(self, name, classes):
    mw = _ModuleWrapper(name, classes)
    self._modules.append(mw)
    return mw