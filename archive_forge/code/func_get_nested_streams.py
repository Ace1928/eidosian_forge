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
def get_nested_streams(dmap):
    """Recurses supplied DynamicMap to find all streams

    Args:
        dmap: DynamicMap to recurse to look for streams

    Returns:
        List of streams that were found
    """
    return list({s for dmap in get_nested_dmaps(dmap) for s in dmap.streams})