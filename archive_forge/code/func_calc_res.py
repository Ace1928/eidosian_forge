from __future__ import annotations
import os
import re
from inspect import getmro
import numba as nb
import numpy as np
import pandas as pd
from toolz import memoize
from xarray import DataArray
import dask.dataframe as dd
import datashader.datashape as datashape
def calc_res(raster):
    """Calculate the resolution of xarray.DataArray raster and return it as the
    two-tuple (xres, yres). yres is positive if it is decreasing.
    """
    h, w = raster.shape[-2:]
    ydim, xdim = raster.dims[-2:]
    xcoords = raster[xdim].values
    ycoords = raster[ydim].values
    xres = (xcoords[-1] - xcoords[0]) / (w - 1)
    yres = (ycoords[0] - ycoords[-1]) / (h - 1)
    return (xres, yres)