import weakref
from .. import functions as fn
from ..graphicsItems.GraphicsObject import GraphicsObject
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
def generatePath(self, start, stop):
    path = QtGui.QPainterPath()
    path.moveTo(start)
    if self.style['shape'] == 'line':
        path.lineTo(stop)
    elif self.style['shape'] == 'cubic':
        path.cubicTo(Point(stop.x(), start.y()), Point(start.x(), stop.y()), Point(stop.x(), stop.y()))
    else:
        raise Exception('Invalid shape "%s"; options are "line" or "cubic"' % self.style['shape'])
    return path