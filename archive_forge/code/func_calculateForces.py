import sys
import weakref
import math
from PySide2 import QtCore, QtGui, QtWidgets
def calculateForces(self):
    if not self.scene() or self.scene().mouseGrabberItem() is self:
        self.newPos = self.pos()
        return
    xvel = 0.0
    yvel = 0.0
    for item in self.scene().items():
        if not isinstance(item, Node):
            continue
        line = QtCore.QLineF(self.mapFromItem(item, 0, 0), QtCore.QPointF(0, 0))
        dx = line.dx()
        dy = line.dy()
        l = 2.0 * (dx * dx + dy * dy)
        if l > 0:
            xvel += dx * 150.0 / l
            yvel += dy * 150.0 / l
    weight = (len(self.edgeList) + 1) * 10.0
    for edge in self.edgeList:
        if edge().sourceNode() is self:
            pos = self.mapFromItem(edge().destNode(), 0, 0)
        else:
            pos = self.mapFromItem(edge().sourceNode(), 0, 0)
        xvel += pos.x() / weight
        yvel += pos.y() / weight
    if QtCore.qAbs(xvel) < 0.1 and QtCore.qAbs(yvel) < 0.1:
        xvel = yvel = 0.0
    sceneRect = self.scene().sceneRect()
    self.newPos = self.pos() + QtCore.QPointF(xvel, yvel)
    self.newPos.setX(min(max(self.newPos.x(), sceneRect.left() + 10), sceneRect.right() - 10))
    self.newPos.setY(min(max(self.newPos.y(), sceneRect.top() + 10), sceneRect.bottom() - 10))