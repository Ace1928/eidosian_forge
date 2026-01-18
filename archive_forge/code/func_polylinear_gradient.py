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
def polylinear_gradient(colors, n):
    """
    Interpolates the color gradients between a list of hex colors.
    """
    n_out = int(float(n) / (len(colors) - 1))
    gradient = linear_gradient(colors[0], colors[1], n_out)
    if len(colors) == len(gradient):
        return gradient
    for col in range(1, len(colors) - 1):
        next_colors = linear_gradient(colors[col], colors[col + 1], n_out + 1)
        gradient += next_colors[1:] if len(next_colors) > 1 else next_colors
    return gradient