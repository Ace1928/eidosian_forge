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
def outer_fn(*outer_key, **dynkwargs):
    if inner_dynamic:

        def inner_fn(*inner_key, **dynkwargs):
            outer_vals = zip(outer_kdims, util.wrap_tuple(outer_key))
            inner_vals = zip(inner_kdims, util.wrap_tuple(inner_key))
            inner_sel = [(k.name, v) for k, v in inner_vals]
            outer_sel = [(k.name, v) for k, v in outer_vals]
            return self.select(**dict(inner_sel + outer_sel))
        return self.clone([], callback=inner_fn, kdims=inner_kdims)
    else:
        dim_vals = [(d.name, d.values) for d in inner_kdims]
        dim_vals += [(d.name, [v]) for d, v in zip(outer_kdims, util.wrap_tuple(outer_key))]
        with item_check(False):
            selected = HoloMap(self.select(**dict(dim_vals)))
            return group_type(selected.reindex(inner_kdims))