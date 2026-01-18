from collections import OrderedDict
from itertools import zip_longest
import logging
import warnings
import numpy as np
import rasterio
from rasterio.io import DatasetReader
from rasterio.transform import guard_transform
def get_plt():
    """import matplotlib.pyplot
    raise import error if matplotlib is not installed
    """
    try:
        import matplotlib.pyplot as plt
        return plt
    except (ImportError, RuntimeError):
        msg = 'Could not import matplotlib\n'
        msg += 'matplotlib required for plotting functions'
        raise ImportError(msg)