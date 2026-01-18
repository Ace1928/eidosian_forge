import sys
import types
import numpy as np
import pandas as pd
from .. import util
from ..dimension import Dimension, asdim, dimension_name
from ..element import Element
from ..ndmapping import NdMapping, item_check, sorted_context
from .grid import GridInterface
from .interface import DataError, Interface
from .util import dask_array_module, finite_range
def retrieve_unit_and_label(dim):
    if isinstance(dim, Dimension):
        return dim
    dim = asdim(dim)
    coord = data[dim.name]
    unit = coord.attrs.get('units') if dim.unit is None else dim.unit
    if isinstance(unit, tuple):
        unit = unit[0]
    if isinstance(coord.attrs.get('long_name'), str):
        spec = (dim.name, coord.attrs['long_name'])
    else:
        spec = (dim.name, dim.label)
    nodata = coord.attrs.get('NODATA')
    return dim.clone(spec, unit=unit, nodata=nodata)