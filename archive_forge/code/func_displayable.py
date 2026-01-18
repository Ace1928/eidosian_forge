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
def displayable(obj):
    """
    Predicate that returns whether the object is displayable or not
    (i.e. whether the object obeys the nesting hierarchy)
    """
    if isinstance(obj, Overlay) and any((isinstance(o, (HoloMap, GridSpace, AdjointLayout)) for o in obj)):
        return False
    if isinstance(obj, HoloMap):
        return obj.type not in [Layout, GridSpace, NdLayout, DynamicMap]
    if isinstance(obj, (GridSpace, Layout, NdLayout)):
        for el in obj.values():
            if not displayable(el):
                return False
        return True
    return True