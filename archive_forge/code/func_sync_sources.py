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
def sync_sources(self):
    """
        Syncs data sources between Elements, which draw data
        from the same object.
        """
    get_sources = lambda x: (id(x.current_frame.data), x)
    filter_fn = lambda x: x.shared_datasource and x.current_frame is not None and (not isinstance(x.current_frame.data, np.ndarray)) and ('source' in x.handles)
    data_sources = self.traverse(get_sources, [filter_fn])
    grouped_sources = groupby(sorted(data_sources, key=lambda x: x[0]), lambda x: x[0])
    shared_sources = []
    source_cols = {}
    plots = []
    for _, group in grouped_sources:
        group = list(group)
        if len(group) > 1:
            source_data = {}
            for _, plot in group:
                source_data.update(plot.handles['source'].data)
            new_source = ColumnDataSource(source_data)
            for _, plot in group:
                renderer = plot.handles.get('glyph_renderer')
                for callback in plot.callbacks:
                    callback.reset()
                if renderer is None:
                    continue
                elif 'data_source' in renderer.properties():
                    renderer.update(data_source=new_source)
                else:
                    renderer.update(source=new_source)
                plot.handles['source'] = plot.handles['cds'] = new_source
                plots.append(plot)
            shared_sources.append(new_source)
            source_cols[id(new_source)] = [c for c in new_source.data]
    for plot in plots:
        for hook in plot.hooks:
            hook(plot, plot.current_frame)
        for callback in plot.callbacks:
            callback.initialize(plot_id=self.id)
    self.handles['shared_sources'] = shared_sources
    self.handles['source_cols'] = source_cols