from contextlib import contextmanager
import logging
import os
import math
from pathlib import Path
import warnings
import numpy as np
import rasterio
from rasterio.coords import disjoint_bounds
from rasterio.enums import Resampling
from rasterio.errors import RasterioDeprecationWarning
from rasterio import windows
from rasterio.transform import Affine
def copy_min(merged_data, new_data, merged_mask, new_mask, **kwargs):
    """Returns the minimum value pixel."""
    mask = np.empty_like(merged_mask, dtype='bool')
    np.logical_or(merged_mask, new_mask, out=mask)
    np.logical_not(mask, out=mask)
    np.minimum(merged_data, new_data, out=merged_data, where=mask, casting='unsafe')
    np.logical_not(new_mask, out=mask)
    np.logical_and(merged_mask, mask, out=mask)
    np.copyto(merged_data, new_data, where=mask, casting='unsafe')