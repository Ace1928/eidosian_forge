import importlib
import warnings
from typing import Any, Dict
import matplotlib as mpl
import numpy as np
import packaging
from matplotlib.colors import to_hex
from scipy.stats import mode, rankdata
from scipy.interpolate import CubicSpline
from ..rcparams import rcParams
from ..stats.density_utils import kde
from ..stats import hdi
def color_from_dim(dataarray, dim_name):
    """Return colors and color mapping of a DataArray using coord values as color code.

    Parameters
    ----------
    dataarray : xarray.DataArray
    dim_name : str
    dimension whose coordinates will be used as color code.

    Returns
    -------
    colors : array of floats
        Array of colors (as floats for use with a cmap) for each element in the dataarray.
    color_mapping : mapping coord_value -> float
        Mapping from coord values to corresponding color
    """
    present_dims = dataarray.dims
    coord_values = dataarray[dim_name].values
    unique_coords = set(coord_values)
    color_mapping = {coord: num / len(unique_coords) for num, coord in enumerate(unique_coords)}
    if len(present_dims) > 1:
        multi_coords = dataarray.coords.to_index()
        coord_idx = present_dims.index(dim_name)
        colors = [color_mapping[coord[coord_idx]] for coord in multi_coords]
    else:
        colors = [color_mapping[coord] for coord in coord_values]
    return (colors, color_mapping)