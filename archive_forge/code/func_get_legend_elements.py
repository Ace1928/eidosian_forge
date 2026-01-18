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
def get_legend_elements(self):
    """
        Return legend elements to display the color code for each category.
        """
    if not isinstance(self._color_key, dict):
        binned = self.get_ds_data()
        categories = binned.coords[binned.dims[2]].data
        color_dict = dict(zip(categories, self._color_key))
    else:
        color_dict = self._color_key
    return [Patch(facecolor=color, edgecolor='none', label=category) for category, color in color_dict.items()]