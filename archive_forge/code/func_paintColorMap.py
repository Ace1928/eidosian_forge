import importlib.util
import re
import numpy as np
from .. import colormap
from .. import functions as fn
from ..graphicsItems.GradientPresets import Gradients
from ..Qt import QtCore, QtGui, QtWidgets
def paintColorMap(self, painter, rect):
    painter.save()
    image = self.getImage()
    painter.drawImage(rect, image)
    if not self.horizontal:
        painter.translate(rect.center())
        painter.rotate(-90)
        painter.translate(-rect.center())
    text = self.colorMap().name
    wpen = QtGui.QPen(QtCore.Qt.GlobalColor.white)
    bpen = QtGui.QPen(QtCore.Qt.GlobalColor.black)
    lightness = image.pixelColor(image.rect().center()).lightnessF()
    if lightness >= 0.5:
        pens = [wpen, bpen]
    else:
        pens = [bpen, wpen]
    AF = QtCore.Qt.AlignmentFlag
    trect = painter.boundingRect(rect, AF.AlignCenter, text)
    painter.setPen(pens[0])
    painter.drawText(trect, 0, text)
    painter.setPen(pens[1])
    painter.drawText(trect.adjusted(1, 0, 1, 0), 0, text)
    painter.restore()