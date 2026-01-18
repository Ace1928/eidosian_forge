from PySide2 import QtCore, QtGui, QtWidgets
import dragdroprobot_rc
def dragLeaveEvent(self, event):
    self.dragOver = False
    self.update()