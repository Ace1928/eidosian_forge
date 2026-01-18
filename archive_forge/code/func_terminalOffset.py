import sys
from collections import OrderedDict
from .. import functions as fn
from ..debug import printExc
from ..graphicsItems.GraphicsObject import GraphicsObject
from ..Qt import QtCore, QtGui, QtWidgets
from .Terminal import Terminal
def terminalOffset(self):
    """
        This method returns the current terminal offset in use.

        :returns: The offset in px.
        """
    return self._nodeOffset