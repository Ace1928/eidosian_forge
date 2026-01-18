from OpenGL.GL import *  # noqa
import numpy as np
from ..MeshData import MeshData
from .GLMeshItem import GLMeshItem
def generateFaces(self):
    cols = self._z.shape[1] - 1
    rows = self._z.shape[0] - 1
    faces = np.empty((cols * rows * 2, 3), dtype=np.uint)
    rowtemplate1 = np.arange(cols).reshape(cols, 1) + np.array([[0, 1, cols + 1]])
    rowtemplate2 = np.arange(cols).reshape(cols, 1) + np.array([[cols + 1, 1, cols + 2]])
    for row in range(rows):
        start = row * cols * 2
        faces[start:start + cols] = rowtemplate1 + row * (cols + 1)
        faces[start + cols:start + cols * 2] = rowtemplate2 + row * (cols + 1)
    self._faces = faces