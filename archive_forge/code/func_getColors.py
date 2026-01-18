from collections.abc import Callable, Sequence
from os import listdir, path
import numpy as np
from .functions import clip_array, clip_scalar, colorDistance, eq, mkColor
from .Qt import QtCore, QtGui
def getColors(self, mode=BYTE):
    """
        Returns a list of the colors associated with the stops of the color map.
        
        The parameter `mode` can be one of
            - `ColorMap.BYTE` or 'byte' to return colors as RGBA tuples in byte format (0 to 255)
            - `ColorMap.FLOAT` or 'float' to return colors as RGBA tuples in float format (0.0 to 1.0)
            - `ColorMap.QCOLOR` or 'qcolor' to return a list of QColors
            
        The default is byte format.
        """
    stops, color = self.getStops(mode=mode)
    return color