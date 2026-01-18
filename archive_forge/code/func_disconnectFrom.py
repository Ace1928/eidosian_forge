import weakref
from .. import functions as fn
from ..graphicsItems.GraphicsObject import GraphicsObject
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
def disconnectFrom(self, term):
    if not self.connectedTo(term):
        return
    item = self._connections[term]
    item.close()
    del self._connections[term]
    del term._connections[self]
    self.recolor()
    term.recolor()
    self.disconnected(term)
    term.disconnected(self)