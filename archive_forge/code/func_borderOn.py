from ..Qt import QtCore, QtWidgets
def borderOn(self):
    self.setStyleSheet(self.indStyle, temporary=True)
    if self.limitedTime or self.count <= 2:
        QtCore.QTimer.singleShot(100, self.borderOff)