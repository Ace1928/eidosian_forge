from collections import defaultdict
import numpy as np
import param
from bokeh.models import (
from ...core.data import Dataset
from ...core.options import Cycle, abbreviated_exception
from ...core.util import dimension_sanitizer, unique_array
from ...util.transform import dim
from ..mixins import ChordMixin, GraphMixin
from ..util import get_directed_graph_paths, process_cmap
from .chart import ColorbarPlot, PointPlot
from .element import CompositeElementPlot, LegendPlot
from .styles import (
@property
def edge_glyph(self):
    if self.filled:
        return 'patches_1'
    elif self.bezier:
        return 'bezier_1'
    else:
        return 'multi_line_1'