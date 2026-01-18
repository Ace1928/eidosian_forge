from PySide2 import QtCore, QtGui, QtWidgets
import dragdroprobot_rc
def dragEnterEvent(self, event):
    if event.mimeData().hasColor() or (isinstance(self, RobotHead) and event.mimeData().hasImage()):
        event.setAccepted(True)
        self.dragOver = True
        self.update()
    else:
        event.setAccepted(False)