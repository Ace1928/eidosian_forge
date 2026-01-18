from PySide2.QtCore import Qt, QSize
from PySide2.QtWidgets import (QApplication, QDialog, QLayout, QGridLayout,
def createRotableGroupBox(self):
    self.rotableGroupBox = QGroupBox('Rotable Widgets')
    self.rotableWidgets.append(QSpinBox())
    self.rotableWidgets.append(QSlider())
    self.rotableWidgets.append(QDial())
    self.rotableWidgets.append(QProgressBar())
    count = len(self.rotableWidgets)
    for i in range(count):
        self.rotableWidgets[i].valueChanged[int].connect(self.rotableWidgets[(i + 1) % count].setValue)
    self.rotableLayout = QGridLayout()
    self.rotableGroupBox.setLayout(self.rotableLayout)
    self.rotateWidgets()