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
def isrealfloat(dt):
    """Check if a datashape is numeric and real.

    Example
    -------
    >>> isrealfloat('int32')
    False
    >>> isrealfloat('float64')
    True
    >>> isrealfloat('string')
    False
    >>> isrealfloat('complex64')
    False
    """
    dt = datashape.predicates.launder(dt)
    return isinstance(dt, datashape.Unit) and dt in datashape.typesets.floating