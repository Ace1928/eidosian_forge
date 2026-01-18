from PySide2.QtCore import Qt, QSize
from PySide2.QtWidgets import (QApplication, QDialog, QLayout, QGridLayout,
def createOptionsGroupBox(self):
    self.optionsGroupBox = QGroupBox('Options')
    buttonsOrientationLabel = QLabel('Orientation of buttons:')
    buttonsOrientationComboBox = QComboBox()
    buttonsOrientationComboBox.addItem('Horizontal', Qt.Horizontal)
    buttonsOrientationComboBox.addItem('Vertical', Qt.Vertical)
    buttonsOrientationComboBox.currentIndexChanged[int].connect(self.buttonsOrientationChanged)
    self.buttonsOrientationComboBox = buttonsOrientationComboBox
    optionsLayout = QGridLayout()
    optionsLayout.addWidget(buttonsOrientationLabel, 0, 0)
    optionsLayout.addWidget(self.buttonsOrientationComboBox, 0, 1)
    optionsLayout.setColumnStretch(2, 1)
    self.optionsGroupBox.setLayout(optionsLayout)