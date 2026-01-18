import importlib.util
import re
import numpy as np
from .. import colormap
from .. import functions as fn
from ..graphicsItems.GradientPresets import Gradients
from ..Qt import QtCore, QtGui, QtWidgets
def buildMenuEntryWidget(cmap, text):
    lut = cmap.getLookupTable(nPts=32, alpha=True)
    qimg = QtGui.QImage(lut, len(lut), 1, QtGui.QImage.Format.Format_RGBA8888)
    pixmap = QtGui.QPixmap.fromImage(qimg)
    widget = QtWidgets.QWidget()
    layout = QtWidgets.QHBoxLayout(widget)
    layout.setContentsMargins(1, 1, 1, 1)
    label1 = QtWidgets.QLabel()
    label1.setScaledContents(True)
    label1.setPixmap(pixmap)
    label2 = QtWidgets.QLabel(text)
    layout.addWidget(label1, 0)
    layout.addWidget(label2, 1)
    return widget