import numpy as np
from ..Qt import QtGui
def setFaces(self, faces):
    """Set the (Nf, 3) array of faces. Each rown in the array contains
        three indexes into the vertex array, specifying the three corners 
        of a triangular face."""
    self._faces = faces
    self._edges = None
    self._vertexFaces = None
    self._vertexesIndexedByFaces = None
    self.resetNormals()
    self._vertexColorsIndexedByFaces = None
    self._faceColorsIndexedByFaces = None