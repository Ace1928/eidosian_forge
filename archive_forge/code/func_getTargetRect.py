import os
import re
from ..GraphicsScene import GraphicsScene
from ..Qt import QtCore, QtWidgets
from ..widgets.FileDialog import FileDialog
def getTargetRect(self):
    if isinstance(self.item, GraphicsScene):
        return self.item.getViewWidget().rect()
    else:
        return self.item.mapRectToDevice(self.item.boundingRect())