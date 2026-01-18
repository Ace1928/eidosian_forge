import itertools
import types
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from itertools import groupby
from numbers import Number
from types import FunctionType
import numpy as np
import param
from ..streams import Params, Stream, streams_list_from_dict
from . import traversal, util
from .accessors import Opts, Redim
from .dimension import Dimension, ViewableElement
from .layout import AdjointLayout, Empty, Layout, Layoutable, NdLayout
from .ndmapping import NdMapping, UniformNdMapping, item_check
from .options import Store, StoreOptions
from .overlay import CompositeOverlay, NdOverlay, Overlay, Overlayable
def get_nested_dmaps(dmap):
    """Recurses DynamicMap to find DynamicMaps inputs

    Args:
        dmap: DynamicMap to recurse to look for DynamicMap inputs

    Returns:
        List of DynamicMap instances that were found
    """
    if not isinstance(dmap, DynamicMap):
        return []
    dmaps = [dmap]
    for o in dmap.callback.inputs:
        dmaps.extend(get_nested_dmaps(o))
    return list(set(dmaps))