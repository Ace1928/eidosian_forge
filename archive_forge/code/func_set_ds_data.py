import warnings
from matplotlib.image import _ImageBase
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox, TransformedBbox, BboxTransform
import matplotlib as mpl
import numpy as np
from . import reductions
from . import transfer_functions as tf
from .colors import Sets1to3
from .core import bypixel, Canvas
def set_ds_data(self, binned):
    """
        Set the aggregate data for the bounding box currently displayed.
        Should be a :class:`xarray.DataArray`.
        """
    self._ds_data = binned