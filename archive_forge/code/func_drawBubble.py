import sys
import math, random
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from PySide2.QtOpenGL import *
def drawBubble(self, painter):
    painter.save()
    painter.translate(self.position.x() - self.radius, self.position.y() - self.radius)
    painter.setBrush(self.brush)
    painter.drawEllipse(0, 0, int(2 * self.radius), int(2 * self.radius))
    painter.restore()