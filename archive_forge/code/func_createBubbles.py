import sys
import math, random
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from PySide2.QtOpenGL import *
def createBubbles(self, number):
    for i in range(number):
        position = QPointF(self.width() * (0.1 + 0.8 * random.random()), self.height() * (0.1 + 0.8 * random.random()))
        radius = min(self.width(), self.height()) * (0.0125 + 0.0875 * random.random())
        velocity = QPointF(self.width() * 0.0125 * (-0.5 + random.random()), self.height() * 0.0125 * (-0.5 + random.random()))
        self.bubbles.append(Bubble(position, radius, velocity))