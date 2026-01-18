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
def contextMenuClicked(self, action):
    if action in [self.rgbAction, self.hsvAction]:
        return
    name, source = action.data()
    if source == 'preset-gradient':
        self.loadPreset(name)
    else:
        if name is None:
            cmap = colormap.ColorMap(None, [0.0, 1.0])
        else:
            cmap = colormap.get(name, source=source)
        self.setColorMap(cmap)
        self.showTicks(False)