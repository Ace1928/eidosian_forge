from PySide2 import QtCore, QtGui, QtWidgets
def createButton(self, text, member):
    button = QtWidgets.QPushButton(text)
    button.clicked.connect(member)
    return button