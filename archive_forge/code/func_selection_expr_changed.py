from collections import namedtuple
import numpy as np
import param
from param.parameterized import bothmethod
from .core.data import Dataset
from .core.element import Element, Layout
from .core.layout import AdjointLayout
from .core.options import CallbackError, Store
from .core.overlay import NdOverlay, Overlay
from .core.spaces import GridSpace
from .streams import (
from .util import DynamicMap
def selection_expr_changed(*_):
    new_selection_expr = inst._cross_filter_stream.selection_expr
    if repr(inst.selection_expr) != repr(new_selection_expr):
        inst.selection_expr = new_selection_expr