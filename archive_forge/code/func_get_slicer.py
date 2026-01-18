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
def get_slicer(start, end):

    def slicer(cell):
        return start <= cell.point < end
    return slicer