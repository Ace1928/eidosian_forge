from PySide2 import QtCore, QtGui, QtWidgets
import states_rc
def boundingRect(self):
    return QtCore.QRectF(QtCore.QPointF(0, 0), QtCore.QSizeF(self.p.size()))