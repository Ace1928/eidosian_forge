import warnings
import numpy as np
from .minc1 import Minc1File, Minc1Image, MincError, MincHeader
def get_scaled_data(self, sliceobj=()):
    """Return scaled data for slice definition `sliceobj`

        Parameters
        ----------
        sliceobj : tuple, optional
            slice definition. If not specified, return whole array

        Returns
        -------
        scaled_arr : array
            array from minc file with scaling applied
        """
    if sliceobj == ():
        raw_data = np.asanyarray(self._image)
    else:
        try:
            raw_data = self._image[sliceobj]
        except (ValueError, TypeError):
            raw_data = np.asanyarray(self._image)[sliceobj]
        else:
            raw_data = np.asanyarray(raw_data)
    return self._normalize(raw_data, sliceobj)