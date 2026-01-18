import warnings
from collections.abc import Callable, Iterable
from functools import partial
import dask.dataframe as dd
import datashader as ds
import datashader.reductions as rd
import datashader.transfer_functions as tf
import numpy as np
import pandas as pd
import param
import xarray as xr
from datashader.colors import color_lookup
from packaging.version import Version
from param.parameterized import bothmethod
from ..core import (
from ..core.data import (
from ..core.util import (
from ..element import (
from ..element.util import connect_tri_edges_pd
from ..streams import PointerXY
from .resample import LinkableOperation, ResampleOperation2D
@classmethod
def rgb2hex(cls, rgb):
    """
        Convert RGB(A) tuple to hex.
        """
    if len(rgb) > 3:
        rgb = rgb[:-1]
    return '#{:02x}{:02x}{:02x}'.format(*(int(v * 255) for v in rgb))