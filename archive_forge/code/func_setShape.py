from PySide2.QtCore import QPoint, QRect, QSize, Qt, qVersion
from PySide2.QtGui import (QBrush, QConicalGradient, QLinearGradient, QPainter,
from PySide2.QtWidgets import (QApplication, QCheckBox, QComboBox, QGridLayout,
import basicdrawing_rc
def setShape(self, shape):
    self.shape = shape
    self.update()