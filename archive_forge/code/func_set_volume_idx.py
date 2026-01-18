import weakref
import numpy as np
from .affines import voxel_sizes
from .optpkg import optional_package
from .orientations import aff2axcodes, axcodes2ornt
def set_volume_idx(self, v):
    """Set current displayed volume index

        Parameters
        ----------
        v : int
            Volume index.
        """
    self._set_volume_index(v)
    self._draw()