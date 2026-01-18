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
def get_minimum_span(low, high, span):
    """
    If lower and high values are equal ensures they are separated by
    the defined span.
    """
    if is_number(low) and low == high:
        if isinstance(low, np.datetime64):
            span = span * np.timedelta64(1, 's')
        low, high = (low - span, high + span)
    return (low, high)