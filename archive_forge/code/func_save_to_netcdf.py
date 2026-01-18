import os
import xarray as xr
import datashader as ds
import pandas as pd
import numpy as np
import pytest
def save_to_netcdf(img, filename):
    """Save a shaded image as NetCDF file.

    Parameters
    ----------
    img: xarray
      The image to save
    filename: unicode
      The name of the file to save to, 'nc' extension will be appended.

    """
    filename = os.path.join(datadir, filename + '.nc')
    print('Saving: ' + filename)
    img.to_netcdf(filename)