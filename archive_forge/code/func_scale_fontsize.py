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
def scale_fontsize(size, scaling):
    """
    Scales a numeric or string font size.
    """
    ext = None
    if isinstance(size, str):
        match = re.match('[-+]?\\d*\\.\\d+|\\d+', size)
        if match:
            value = match.group()
            ext = size.replace(value, '')
            size = float(value)
        else:
            return size
    if scaling:
        size = size * scaling
    if ext is not None:
        size = f'{size:.3f}'.rstrip('0').rstrip('.') + ext
    return size