import sys
import math
from PySide2 import QtCore, QtGui, QtWidgets, QtOpenGL
def makeGear(self, reflectance, innerRadius, outerRadius, thickness, toothSize, toothCount):
    list = glGenLists(1)
    glNewList(list, GL_COMPILE)
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, reflectance)
    r0 = innerRadius
    r1 = outerRadius - toothSize / 2.0
    r2 = outerRadius + toothSize / 2.0
    delta = 2.0 * math.pi / toothCount / 4.0
    z = thickness / 2.0
    glShadeModel(GL_FLAT)
    for i in range(2):
        if i == 0:
            sign = +1.0
        else:
            sign = -1.0
        glNormal3d(0.0, 0.0, sign)
        glBegin(GL_QUAD_STRIP)
        for j in range(toothCount + 1):
            angle = 2.0 * math.pi * j / toothCount
            glVertex3d(r0 * math.cos(angle), r0 * math.sin(angle), sign * z)
            glVertex3d(r1 * math.cos(angle), r1 * math.sin(angle), sign * z)
            glVertex3d(r0 * math.cos(angle), r0 * math.sin(angle), sign * z)
            glVertex3d(r1 * math.cos(angle + 3 * delta), r1 * math.sin(angle + 3 * delta), sign * z)
        glEnd()
        glBegin(GL_QUADS)
        for j in range(toothCount):
            angle = 2.0 * math.pi * j / toothCount
            glVertex3d(r1 * math.cos(angle), r1 * math.sin(angle), sign * z)
            glVertex3d(r2 * math.cos(angle + delta), r2 * math.sin(angle + delta), sign * z)
            glVertex3d(r2 * math.cos(angle + 2 * delta), r2 * math.sin(angle + 2 * delta), sign * z)
            glVertex3d(r1 * math.cos(angle + 3 * delta), r1 * math.sin(angle + 3 * delta), sign * z)
        glEnd()
    glBegin(GL_QUAD_STRIP)
    for i in range(toothCount):
        for j in range(2):
            angle = 2.0 * math.pi * (i + j / 2.0) / toothCount
            s1 = r1
            s2 = r2
            if j == 1:
                s1, s2 = (s2, s1)
            glNormal3d(math.cos(angle), math.sin(angle), 0.0)
            glVertex3d(s1 * math.cos(angle), s1 * math.sin(angle), +z)
            glVertex3d(s1 * math.cos(angle), s1 * math.sin(angle), -z)
            glNormal3d(s2 * math.sin(angle + delta) - s1 * math.sin(angle), s1 * math.cos(angle) - s2 * math.cos(angle + delta), 0.0)
            glVertex3d(s2 * math.cos(angle + delta), s2 * math.sin(angle + delta), +z)
            glVertex3d(s2 * math.cos(angle + delta), s2 * math.sin(angle + delta), -z)
    glVertex3d(r1, 0.0, +z)
    glVertex3d(r1, 0.0, -z)
    glEnd()
    glShadeModel(GL_SMOOTH)
    glBegin(GL_QUAD_STRIP)
    for i in range(toothCount + 1):
        angle = i * 2.0 * math.pi / toothCount
        glNormal3d(-math.cos(angle), -math.sin(angle), 0.0)
        glVertex3d(r0 * math.cos(angle), r0 * math.sin(angle), +z)
        glVertex3d(r0 * math.cos(angle), r0 * math.sin(angle), -z)
    glEnd()
    glEndList()
    return list