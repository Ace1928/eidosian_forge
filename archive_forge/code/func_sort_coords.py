import sys
import datetime
from itertools import product
import numpy as np
from holoviews.core.data import Dataset
from holoviews.core.data.interface import Interface, DataError
from holoviews.core.data.grid import GridInterface
from holoviews.core.dimension import Dimension, asdim
from holoviews.core.element import Element
from holoviews.core.ndmapping import (NdMapping, item_check, sorted_context)
from holoviews.core.spaces import HoloMap
from holoviews.core import util
def sort_coords(coord):
    """
    Sorts a list of DimCoords trying to ensure that
    dates and pressure levels appear first and the
    longitude and latitude appear last in the correct
    order.
    """
    import iris
    order = {'T': -2, 'Z': -1, 'X': 1, 'Y': 2}
    axis = iris.util.guess_coord_axis(coord)
    return (order.get(axis, 0), coord and coord.name())