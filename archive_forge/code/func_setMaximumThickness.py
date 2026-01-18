import importlib.util
import re
import numpy as np
from .. import colormap
from .. import functions as fn
from ..graphicsItems.GradientPresets import Gradients
from ..Qt import QtCore, QtGui, QtWidgets
def setMaximumThickness(self, val):
    Thickness = 'Height' if self.horizontal else 'Width'
    getattr(self, f'setMaximum{Thickness}')(val)