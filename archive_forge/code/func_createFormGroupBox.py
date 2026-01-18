from PySide2 import QtCore, QtGui, QtWidgets
def createFormGroupBox(self):
    self.formGroupBox = QtWidgets.QGroupBox('Form layout')
    layout = QtWidgets.QFormLayout()
    layout.addRow(QtWidgets.QLabel('Line 1:'), QtWidgets.QLineEdit())
    layout.addRow(QtWidgets.QLabel('Line 2, long text:'), QtWidgets.QComboBox())
    layout.addRow(QtWidgets.QLabel('Line 3:'), QtWidgets.QSpinBox())
    self.formGroupBox.setLayout(layout)