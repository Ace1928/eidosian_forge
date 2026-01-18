from PySide2.QtCore import Qt, QSize
from PySide2.QtWidgets import (QApplication, QDialog, QLayout, QGridLayout,
def buttonsOrientationChanged(self, index):
    self.mainLayout.setSizeConstraint(QLayout.SetNoConstraint)
    self.setMinimumSize(0, 0)
    orientation = Qt.Orientation(int(self.buttonsOrientationComboBox.itemData(index)))
    if orientation == self.buttonBox.orientation():
        return
    self.mainLayout.removeWidget(self.buttonBox)
    spacing = self.mainLayout.spacing()
    oldSizeHint = self.buttonBox.sizeHint() + QSize(spacing, spacing)
    self.buttonBox.setOrientation(orientation)
    newSizeHint = self.buttonBox.sizeHint() + QSize(spacing, spacing)
    if orientation == Qt.Horizontal:
        self.mainLayout.addWidget(self.buttonBox, 2, 0)
        self.resize(self.size() + QSize(-oldSizeHint.width(), newSizeHint.height()))
    else:
        self.mainLayout.addWidget(self.buttonBox, 0, 3, 2, 1)
        self.resize(self.size() + QSize(newSizeHint.width(), -oldSizeHint.height()))
    self.mainLayout.setSizeConstraint(QLayout.SetDefaultConstraint)