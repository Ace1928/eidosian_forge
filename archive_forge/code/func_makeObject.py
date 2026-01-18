import sys
import math
from PySide2 import QtCore, QtGui, QtWidgets
def makeObject(self):
    genList = GL.glGenLists(1)
    GL.glNewList(genList, GL.GL_COMPILE)
    GL.glBegin(GL.GL_QUADS)
    x1 = +0.06
    y1 = -0.14
    x2 = +0.14
    y2 = -0.06
    x3 = +0.08
    y3 = +0.0
    x4 = +0.3
    y4 = +0.22
    self.quad(x1, y1, x2, y2, y2, x2, y1, x1)
    self.quad(x3, y3, x4, y4, y4, x4, y3, x3)
    self.extrude(x1, y1, x2, y2)
    self.extrude(x2, y2, y2, x2)
    self.extrude(y2, x2, y1, x1)
    self.extrude(y1, x1, x1, y1)
    self.extrude(x3, y3, x4, y4)
    self.extrude(x4, y4, y4, x4)
    self.extrude(y4, x4, y3, x3)
    Pi = 3.141592653589793
    NumSectors = 200
    for i in range(NumSectors):
        angle1 = i * 2 * Pi / NumSectors
        x5 = 0.3 * math.sin(angle1)
        y5 = 0.3 * math.cos(angle1)
        x6 = 0.2 * math.sin(angle1)
        y6 = 0.2 * math.cos(angle1)
        angle2 = (i + 1) * 2 * Pi / NumSectors
        x7 = 0.2 * math.sin(angle2)
        y7 = 0.2 * math.cos(angle2)
        x8 = 0.3 * math.sin(angle2)
        y8 = 0.3 * math.cos(angle2)
        self.quad(x5, y5, x6, y6, x7, y7, x8, y8)
        self.extrude(x6, y6, x7, y7)
        self.extrude(x8, y8, x5, y5)
    GL.glEnd()
    GL.glEndList()
    return genList