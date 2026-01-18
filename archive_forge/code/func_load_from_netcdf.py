import os
import xarray as xr
import datashader as ds
import pandas as pd
import numpy as np
import pytest
def load_from_netcdf(filename):
    """Load a shaded image from NetCDF file.

    Parameters
    ----------
    filename: unicode
      The name of the file to load from.

    Returns
    -------
    img: xarray
      The loaded image.

    """
    filename = os.path.join(datadir, filename + '.nc')
    return xr.open_dataarray(filename)