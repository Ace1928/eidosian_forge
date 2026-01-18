from PySide2.QtCore import (QFile, QFileInfo, QPoint, QSettings, QSignalMapper,
from PySide2.QtGui import QIcon, QKeySequence
from PySide2.QtWidgets import (QAction, QApplication, QFileDialog, QMainWindow,
import mdi_rc
def createMdiChild(self):
    child = MdiChild()
    self.mdiArea.addSubWindow(child)
    child.copyAvailable.connect(self.cutAct.setEnabled)
    child.copyAvailable.connect(self.copyAct.setEnabled)
    return child