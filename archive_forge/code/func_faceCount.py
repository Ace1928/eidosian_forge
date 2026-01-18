import numpy as np
from ..Qt import QtGui
def faceCount(self):
    """
        Return the number of faces in the mesh.
        """
    if self._faces is not None:
        return self._faces.shape[0]
    elif self._vertexesIndexedByFaces is not None:
        return self._vertexesIndexedByFaces.shape[0]