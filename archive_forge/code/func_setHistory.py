from ..Qt import QtCore, QtWidgets
def setHistory(self, num):
    self.ptr = num
    self.setText(self.history[self.ptr])