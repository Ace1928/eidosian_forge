import operator
import weakref
import numpy as np
from .. import functions as fn
from .. import colormap
from ..colormap import ColorMap
from ..Qt import QtCore, QtGui, QtWidgets
from ..widgets.SpinBox import SpinBox
from ..widgets.ColorMapButton import ColorMapMenu
from .GraphicsWidget import GraphicsWidget
from .GradientPresets import Gradients
def listTicks(self):
    """Return a sorted list of all the Tick objects on the slider."""
    ticks = sorted(self.ticks.items(), key=operator.itemgetter(1))
    return ticks