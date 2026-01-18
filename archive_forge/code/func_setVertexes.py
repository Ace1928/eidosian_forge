import numpy as np
from ..Qt import QtGui
def setVertexes(self, verts=None, indexed=None, resetNormals=True):
    """
        Set the array (Nv, 3) of vertex coordinates.
        If indexed=='faces', then the data must have shape (Nf, 3, 3) and is
        assumed to be already indexed as a list of faces.
        This will cause any pre-existing normal vectors to be cleared
        unless resetNormals=False.
        """
    if indexed is None:
        if verts is not None:
            self._vertexes = np.ascontiguousarray(verts, dtype=np.float32)
        self._vertexesIndexedByFaces = None
    elif indexed == 'faces':
        self._vertexes = None
        if verts is not None:
            self._vertexesIndexedByFaces = np.ascontiguousarray(verts, dtype=np.float32)
    else:
        raise Exception("Invalid indexing mode. Accepts: None, 'faces'")
    if resetNormals:
        self.resetNormals()