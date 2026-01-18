import sys
from importlib.util import find_spec
import numpy as np
import pandas as pd
from ..core import Dataset, NdOverlay, util
from ..streams import Lasso, Selection1D, SelectionXY
from ..util.transform import dim
from .annotation import HSpan, VSpan
def spatial_select(xvals, yvals, geometry):
    if xvals.ndim > 1:
        return spatial_select_gridded(xvals, yvals, geometry)
    else:
        return spatial_select_columnar(xvals, yvals, geometry)