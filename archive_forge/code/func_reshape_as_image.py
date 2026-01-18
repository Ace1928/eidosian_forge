from collections import OrderedDict
from itertools import zip_longest
import logging
import warnings
import numpy as np
import rasterio
from rasterio.io import DatasetReader
from rasterio.transform import guard_transform
def reshape_as_image(arr):
    """Returns the source array reshaped into the order
    expected by image processing and visualization software
    (matplotlib, scikit-image, etc)
    by swapping the axes order from (bands, rows, columns)
    to (rows, columns, bands)

    Parameters
    ----------
    arr : array-like of shape (bands, rows, columns)
        image to reshape
    """
    im = np.ma.transpose(arr, [1, 2, 0])
    return im