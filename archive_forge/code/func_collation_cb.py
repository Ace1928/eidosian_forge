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
def collation_cb(*args, **kwargs):
    layout = self[args]
    layout_type = type(layout).__name__
    if len(container.keys()) != len(layout.keys()):
        raise ValueError('Collated DynamicMaps must return %s with consistent number of items.' % layout_type)
    key = kwargs['selection_key']
    index = kwargs['selection_index']
    obj_type = kwargs['selection_type']
    dyn_type_map = defaultdict(list)
    for k, v in layout.data.items():
        if k == key:
            return layout[k]
        dyn_type_map[type(v)].append(v)
    dyn_type_counter = {t: len(vals) for t, vals in dyn_type_map.items()}
    if dyn_type_counter != type_counter:
        raise ValueError('The objects in a %s returned by a DynamicMap must consistently return the same number of items of the same type.' % layout_type)
    return dyn_type_map[obj_type][index]