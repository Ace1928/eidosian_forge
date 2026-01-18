import uuid
import warnings
from ast import literal_eval
from collections import Counter, defaultdict
from functools import partial
from itertools import groupby, product
import numpy as np
import param
from panel.config import config
from panel.io.document import unlocked
from panel.io.notebook import push
from panel.io.state import state
from pyviz_comms import JupyterComm
from ..core import traversal, util
from ..core.data import Dataset, disable_pipeline
from ..core.element import Element, Element3D
from ..core.layout import Empty, Layout, NdLayout
from ..core.options import Compositor, SkipRendering, Store, lookup_options
from ..core.overlay import CompositeOverlay, NdOverlay, Overlay
from ..core.spaces import DynamicMap, HoloMap
from ..core.util import isfinite, stream_parameters
from ..element import Graph, Table
from ..selection import NoOpSelectionDisplay
from ..streams import RangeX, RangeXY, RangeY, Stream
from ..util.transform import dim
from .util import (
@pane.setter
def pane(self, pane):
    if config.console_output != 'disable' and self.root and (self.root.ref['id'] not in state._handles) and isinstance(self.comm, JupyterComm):
        from IPython.display import display
        handle = display(display_id=uuid.uuid4().hex)
        state._handles[self.root.ref['id']] = (handle, [])
    self._pane = pane
    if self.subplots:
        for plot in self.subplots.values():
            if plot is not None:
                plot.pane = pane
            if plot is None or not plot.root:
                continue
            for cb in getattr(plot, 'callbacks', []):
                if hasattr(pane, '_on_error') and getattr(cb, 'comm', None):
                    cb.comm._on_error = partial(pane._on_error, plot.root.ref['id'])
    elif self.root:
        for cb in getattr(self, 'callbacks', []):
            if hasattr(pane, '_on_error') and getattr(cb, 'comm', None):
                cb.comm._on_error = partial(pane._on_error, self.root.ref['id'])