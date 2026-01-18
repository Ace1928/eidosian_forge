from PySide2 import QtCore, QtGui, QtWidgets
import appchooser_rc
def setGeometry(self, rect):
    super(Pixmap, self).setGeometry(rect)
    if rect.size().width() > self.orig.size().width():
        self.p = self.orig.scaled(rect.size().toSize())
    else:
        self.p = QtGui.QPixmap(self.orig)