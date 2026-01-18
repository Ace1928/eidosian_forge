import warnings
import numpy as np
import param
from packaging.version import Version
from param import _is_number
from ..core import (
from ..core.data import ArrayInterface, DictInterface, PandasInterface, default_datatype
from ..core.data.util import dask_array_module
from ..core.util import (
from ..element.chart import Histogram, Scatter
from ..element.path import Contours, Polygons
from ..element.raster import RGB, Image
from ..element.util import categorical_aggregate2d  # noqa (API import)
from ..streams import RangeXY
from ..util.locator import MaxNLocator
def points_to_datetime(points):
    xs, ys = np.split(points, 2, axis=1)
    if data_is_datetime[0]:
        xs = coords_to_datetime(xs)
    if data_is_datetime[1]:
        ys = coords_to_datetime(ys)
    return np.concatenate((xs, ys), axis=1)