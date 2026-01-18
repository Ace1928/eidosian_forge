import numpy as np
from ..Qt import QtGui
def faceColors(self, indexed=None):
    """
        Return an array (Nf, 4) of face colors.
        If indexed=='faces', then instead return an indexed array
        (Nf, 3, 4)  (note this is just the same array with each color
        repeated three times). 
        """
    if indexed is None:
        return self._faceColors
    elif indexed == 'faces':
        if self._faceColorsIndexedByFaces is None and self._faceColors is not None:
            Nf = self._faceColors.shape[0]
            self._faceColorsIndexedByFaces = np.empty((Nf, 3, 4), dtype=self._faceColors.dtype)
            self._faceColorsIndexedByFaces[:] = self._faceColors.reshape(Nf, 1, 4)
        return self._faceColorsIndexedByFaces
    else:
        raise Exception("Invalid indexing mode. Accepts: None, 'faces'")