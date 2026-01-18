from matplotlib.collections import QuadMesh
import numpy as np
import numpy.ma as ma
from cartopy.mpl import _MPL_38
def set_clim(self, vmin=None, vmax=None):
    if hasattr(self, '_wrapped_collection_fix'):
        self._wrapped_collection_fix.set_clim(vmin, vmax)
    super().set_clim(vmin, vmax)