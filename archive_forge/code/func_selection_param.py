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
def selection_param(self, data):
    """
        Returns a parameter which reflects the current selection
        when applied to the supplied data, making it easy to create
        a callback which depends on the current selection.

        Args:
            data: A Dataset type or data which can be cast to a Dataset

        Returns:
            A parameter which reflects the current selection
        """
    raw = False
    if not isinstance(data, Dataset):
        raw = True
        data = Dataset(data)
    pipe = Pipe(data=data.data)
    self._datasets.append((pipe, data, raw))
    return pipe.param.data