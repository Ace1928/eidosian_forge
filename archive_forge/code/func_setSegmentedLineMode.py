from ..Qt import QtCore, QtGui, QtWidgets
import math
import sys
import warnings
import numpy as np
from .. import Qt, debug
from .. import functions as fn
from .. import getConfigOption
from .GraphicsObject import GraphicsObject
def setSegmentedLineMode(self, mode):
    """
        Sets the mode that decides whether or not lines are drawn as segmented lines. Drawing lines
        as segmented lines is more performant than the standard drawing method with continuous
        lines.

        Parameters
        ----------
        mode : str
               ``'auto'`` (default) segmented lines are drawn if the pen's width > 1, pen style is a
               solid line, the pen color is opaque and anti-aliasing is not enabled.

               ``'on'`` lines are always drawn as segmented lines

               ``'off'`` lines are never drawn as segmented lines, i.e. the drawing
               method with continuous lines is used
        """
    if mode not in ('auto', 'on', 'off'):
        raise ValueError(f'segmentedLineMode must be "auto", "on" or "off", got {mode} instead')
    self.opts['segmentedLineMode'] = mode
    self.invalidateBounds()
    self.update()