import numpy as np
from ..Qt import QtGui
def resetNormals(self):
    self._vertexNormals = None
    self._vertexNormalsIndexedByFaces = None
    self._faceNormals = None
    self._faceNormalsIndexedByFaces = None