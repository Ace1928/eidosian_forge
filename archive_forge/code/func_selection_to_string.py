from itertools import product, tee
import numpy as np
import xarray as xr
from .labels import BaseLabeller
def selection_to_string(selection):
    """Convert dictionary of coordinates to a string for labels.

    Parameters
    ----------
    selection : dict[Any] -> Any

    Returns
    -------
    str
        key1: value1, key2: value2, ...
    """
    return ', '.join([f'{v}' for _, v in selection.items()])