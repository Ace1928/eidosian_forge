import os
import re
from ..GraphicsScene import GraphicsScene
from ..Qt import QtCore, QtWidgets
from ..widgets.FileDialog import FileDialog
def getSourceRect(self):
    if isinstance(self.item, GraphicsScene):
        w = self.item.getViewWidget()
        return w.viewportTransform().inverted()[0].mapRect(w.rect())
    else:
        return self.item.sceneBoundingRect()