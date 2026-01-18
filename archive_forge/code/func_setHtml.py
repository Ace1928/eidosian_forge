from math import atan2, degrees
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from .GraphicsObject import GraphicsObject
def setHtml(self, html):
    """
        Set the HTML code to be rendered by this item. 
        
        See QtWidgets.QGraphicsTextItem.setHtml().
        """
    if self.toHtml() != html:
        self.textItem.setHtml(html)
        self.updateTextPos()