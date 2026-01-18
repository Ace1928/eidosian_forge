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
def compute_ranges(self, obj, key, ranges):
    """
        Given an object, a specific key, and the normalization options,
        this method will find the specified normalization options on
        the appropriate OptionTree, group the elements according to
        the selected normalization option (i.e. either per frame or
        over the whole animation) and finally compute the dimension
        ranges in each group. The new set of ranges is returned.
        """
    prev_frame = getattr(self, 'prev_frame', None)
    all_table = all((isinstance(el, Table) for el in obj.traverse(lambda x: x, [Element])))
    if obj is None or not self.normalize or all_table:
        return {}
    ranges = self.ranges if ranges is None else {k: dict(v) for k, v in ranges.items()}
    norm_opts = self._get_norm_opts(obj)
    return_fn = lambda x: x if isinstance(x, Element) else None
    for group, (axiswise, framewise, robust) in norm_opts.items():
        axiswise = not getattr(self, 'shared_axes', True) or axiswise
        elements = []
        framewise = framewise or self.dynamic or len(elements) == 1
        if not framewise:
            elements = obj.traverse(return_fn, [group])
        elif key is not None:
            frame = self._get_frame(key)
            elements = [] if frame is None else frame.traverse(return_fn, [group])
        if not (axiswise and (not isinstance(obj, HoloMap))) or (not framewise and isinstance(obj, HoloMap)):
            self._compute_group_range(group, elements, ranges, framewise, axiswise, robust, self.top_level, prev_frame)
    self.ranges.update(ranges)
    return ranges