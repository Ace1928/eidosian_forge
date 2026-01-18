from PySide2.QtCore import QPoint, QRect, QSize, Qt, qVersion
from PySide2.QtGui import (QBrush, QConicalGradient, QLinearGradient, QPainter,
from PySide2.QtWidgets import (QApplication, QCheckBox, QComboBox, QGridLayout,
import basicdrawing_rc
def penChanged(self):
    width = self.penWidthSpinBox.value()
    style = Qt.PenStyle(self.penStyleComboBox.itemData(self.penStyleComboBox.currentIndex(), IdRole))
    cap = Qt.PenCapStyle(self.penCapComboBox.itemData(self.penCapComboBox.currentIndex(), IdRole))
    join = Qt.PenJoinStyle(self.penJoinComboBox.itemData(self.penJoinComboBox.currentIndex(), IdRole))
    self.renderArea.setPen(QPen(Qt.blue, width, style, cap, join))