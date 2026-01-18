from PySide2 import QtCore, QtGui, QtWidgets
def createHorizontalGroupBox(self):
    self.horizontalGroupBox = QtWidgets.QGroupBox('Horizontal layout')
    layout = QtWidgets.QHBoxLayout()
    for i in range(Dialog.NumButtons):
        button = QtWidgets.QPushButton('Button %d' % (i + 1))
        layout.addWidget(button)
    self.horizontalGroupBox.setLayout(layout)