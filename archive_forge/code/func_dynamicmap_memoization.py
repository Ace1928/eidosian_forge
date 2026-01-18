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
@contextmanager
def dynamicmap_memoization(callable_obj, streams):
    """
    Determine whether the Callable should have memoization enabled
    based on the supplied streams (typically by a
    DynamicMap). Memoization is disabled if any of the streams require
    it it and are currently in a triggered state.
    """
    memoization_state = bool(callable_obj._stream_memoization)
    callable_obj._stream_memoization &= not any((s.transient and s._triggering for s in streams))
    try:
        yield
    finally:
        callable_obj._stream_memoization = memoization_state