import numpy as np
from ..Qt import QtGui
def hasVertexColor(self):
    """Return True if this data set has vertex color information"""
    for v in (self._vertexColors, self._vertexColorsIndexedByFaces, self._vertexColorsIndexedByEdges):
        if v is not None:
            return True
    return False