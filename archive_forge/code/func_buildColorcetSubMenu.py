import importlib.util
import re
import numpy as np
from .. import colormap
from .. import functions as fn
from ..graphicsItems.GradientPresets import Gradients
from ..Qt import QtCore, QtGui, QtWidgets
def buildColorcetSubMenu(self):
    source = 'colorcet'
    import colorcet
    names = list(colorcet.palette_n.keys())
    self.buildSubMenu(names, source)