import sys
from PySide2 import QtCore, QtGui, QtWidgets, QtOpenGL
import textures_rc
def rotateOneStep(self):
    if self.currentGlWidget:
        self.currentGlWidget.rotateBy(+2 * 16, +2 * 16, -1 * 16)