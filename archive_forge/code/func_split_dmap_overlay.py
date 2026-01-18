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
def split_dmap_overlay(obj, depth=0):
    """
    Splits a DynamicMap into the original component layers it was
    constructed from by traversing the graph to search for dynamically
    overlaid components (i.e. constructed by using * on a DynamicMap).
    Useful for assigning subplots of an OverlayPlot the streams that
    are responsible for driving their updates. Allows the OverlayPlot
    to determine if a stream update should redraw a particular
    subplot.
    """
    layers = []
    if isinstance(obj, DynamicMap):
        initialize_dynamic(obj)
        if issubclass(obj.type, NdOverlay) and (not depth):
            for _ in obj.last.values():
                layers.append(obj)
        elif issubclass(obj.type, Overlay):
            if obj.callback.inputs and is_dynamic_overlay(obj):
                for inp in obj.callback.inputs:
                    layers += split_dmap_overlay(inp, depth + 1)
            else:
                for _ in obj.last.values():
                    layers.append(obj)
        else:
            layers.append(obj)
        return layers
    if isinstance(obj, Overlay):
        for _k, v in obj.items():
            layers.append(v)
    else:
        layers.append(obj)
    return layers