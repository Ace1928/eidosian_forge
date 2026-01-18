import bisect
import re
import traceback
import warnings
from collections import defaultdict, namedtuple
import numpy as np
import param
from packaging.version import Version
from ..core import (
from ..core.ndmapping import item_check
from ..core.operation import Operation
from ..core.options import CallbackError, Cycle
from ..core.spaces import get_nested_streams
from ..core.util import (
from ..element import Points
from ..streams import LinkedStream, Params
from ..util.transform import dim
def get_min_distance(element):
    """
    Gets the minimum sampling distance of the x- and y-coordinates
    in a grid.
    """
    try:
        from scipy.spatial.distance import pdist
        return pdist(element.array([0, 1])).min()
    except Exception:
        return _get_min_distance_numpy(element)