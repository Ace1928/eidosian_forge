import sys
import math
from PySide2 import QtCore, QtGui, QtWidgets, QtOpenGL
def renderIntoPixmap(self):
    size = self.getSize()
    if size.isValid():
        pixmap = self.glWidget.renderPixmap(size.width(), size.height())
        self.setPixmap(pixmap)