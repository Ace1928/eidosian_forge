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
def attach_streams(plot, obj, precedence=1.1):
    """
    Attaches plot refresh to all streams on the object.
    """

    def append_refresh(dmap):
        for stream in get_nested_streams(dmap):
            if plot.refresh not in stream._subscribers:
                stream.add_subscriber(plot.refresh, precedence)
    return obj.traverse(append_refresh, [DynamicMap])