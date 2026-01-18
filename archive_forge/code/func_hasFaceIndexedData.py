import numpy as np
from ..Qt import QtGui
def hasFaceIndexedData(self):
    """Return True if this object already has vertex positions indexed by face"""
    return self._vertexesIndexedByFaces is not None