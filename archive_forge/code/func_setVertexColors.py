import numpy as np
from ..Qt import QtGui
def setVertexColors(self, colors, indexed=None):
    """
        Set the vertex color array (Nv, 4).
        If indexed=='faces', then the array will be interpreted
        as indexed and should have shape (Nf, 3, 4)
        """
    if indexed is None:
        self._vertexColors = np.ascontiguousarray(colors, dtype=np.float32)
        self._vertexColorsIndexedByFaces = None
    elif indexed == 'faces':
        self._vertexColors = None
        self._vertexColorsIndexedByFaces = np.ascontiguousarray(colors, dtype=np.float32)
    else:
        raise Exception("Invalid indexing mode. Accepts: None, 'faces'")