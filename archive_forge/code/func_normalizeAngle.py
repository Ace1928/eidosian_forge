import math
from PySide2 import QtCore, QtGui, QtWidgets
import mice_rc
@staticmethod
def normalizeAngle(angle):
    while angle < 0:
        angle += Mouse.TwoPi
    while angle > Mouse.TwoPi:
        angle -= Mouse.TwoPi
    return angle