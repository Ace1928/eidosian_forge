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
def isoverlay_fn(obj):
    """
    Determines whether object is a DynamicMap returning (Nd)Overlay types.
    """
    return isinstance(obj, DynamicMap) and isinstance(obj.last, CompositeOverlay)