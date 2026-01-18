from collections import defaultdict
from itertools import groupby
import numpy as np
import param
from bokeh.layouts import gridplot
from bokeh.models import (
from bokeh.models.layouts import TabPanel, Tabs
from ...core import (
from ...core.options import SkipRendering
from ...core.util import (
from ...selection import NoOpSelectionDisplay
from ..links import Link
from ..plot import (
from ..util import attach_streams, collate, displayable
from .links import LinkCallback
from .util import (
def init_links(self):
    links = LinkCallback.find_links(self)
    callbacks = []
    for link, src_plot, tgt_plot in links:
        cb = Link._callbacks['bokeh'][type(link)]
        if src_plot is None or (link._requires_target and tgt_plot is None):
            continue
        callbacks.append(cb(self.root, link, src_plot, tgt_plot))
    return callbacks