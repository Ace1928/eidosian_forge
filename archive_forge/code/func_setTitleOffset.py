import sys
from collections import OrderedDict
from .. import functions as fn
from ..debug import printExc
from ..graphicsItems.GraphicsObject import GraphicsObject
from ..Qt import QtCore, QtGui, QtWidgets
from .Terminal import Terminal
def setTitleOffset(self, new_offset):
    """
        This method sets the rendering offset introduced after the title of the node.
        This method automatically updates the terminal labels. The default for this value is 25px.

        :param new_offset: The new offset to use in pixels at 100% scale.
        """
    self._titleOffset = new_offset
    self.updateTerminals()