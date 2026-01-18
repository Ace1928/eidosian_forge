import math
from PySide2 import QtCore, QtGui, QtWidgets
import diagramscene_rc
def createBackgroundCellWidget(self, text, image):
    button = QtWidgets.QToolButton()
    button.setText(text)
    button.setIcon(QtGui.QIcon(image))
    button.setIconSize(QtCore.QSize(50, 50))
    button.setCheckable(True)
    self.backgroundButtonGroup.addButton(button)
    layout = QtWidgets.QGridLayout()
    layout.addWidget(button, 0, 0, QtCore.Qt.AlignHCenter)
    layout.addWidget(QtWidgets.QLabel(text), 1, 0, QtCore.Qt.AlignCenter)
    widget = QtWidgets.QWidget()
    widget.setLayout(layout)
    return widget