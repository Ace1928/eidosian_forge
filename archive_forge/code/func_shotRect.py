import sys
import math
from PySide2 import QtCore, QtGui, QtWidgets
def shotRect(self):
    gravity = 4.0
    time = self.timerCount / 40.0
    velocity = self.shootForce
    radians = self.shootAngle * 3.14159265 / 180
    velx = velocity * math.cos(radians)
    vely = velocity * math.sin(radians)
    x0 = (CannonField.barrelRect.right() + 5) * math.cos(radians)
    y0 = (CannonField.barrelRect.right() + 5) * math.sin(radians)
    x = x0 + velx * time
    y = y0 + vely * time - 0.5 * gravity * time * time
    result = QtCore.QRect(0, 0, 6, 6)
    result.moveCenter(QtCore.QPoint(round(x), self.height() - 1 - round(y)))
    return result