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
@property
def selected_cmap(self):
    """
        The datashader colormap for selected data
        """
    return None if self.selected_color is None else _color_to_cmap(self.selected_color)