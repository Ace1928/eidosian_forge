import numpy as np
from bokeh.layouts import Column
from bokeh.models import Div, Toolbar
from holoviews.core import (
from holoviews.element import Curve, Image, Points
from holoviews.operation import gridmatrix
from holoviews.streams import Stream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_grid_shared_source_synced_update(self):
    hmap = HoloMap({i: Dataset({chr(65 + j): np.random.rand(i + 2) for j in range(4)}, kdims=['A', 'B', 'C', 'D']) for i in range(3)})
    hmap1 = hmap.map(lambda x: Points(x.clone(kdims=['A', 'B'])), Dataset)
    hmap2 = hmap.map(lambda x: Points(x.clone(kdims=['D', 'C'])), Dataset)
    hmap2.pop(1)
    grid = GridSpace({0: hmap1, 2: hmap2}, kdims=['X']).opts(shared_datasource=True)
    plot = bokeh_renderer.get_plot(grid)
    sources = plot.handles.get('shared_sources', [])
    source_cols = plot.handles.get('source_cols', {})
    self.assertEqual(len(sources), 1)
    source = sources[0]
    data = source.data
    cols = source_cols[id(source)]
    self.assertEqual(set(cols), {'A', 'B', 'C', 'D'})
    self.assertEqual(set(data.keys()), {'A', 'B', 'C', 'D'})
    plot.update((1,))
    self.assertEqual(data['A'], hmap1[1].dimension_values(0))
    self.assertEqual(data['B'], hmap1[1].dimension_values(1))
    self.assertEqual(data['C'], np.full_like(hmap1[1].dimension_values(0), np.nan))
    self.assertEqual(data['D'], np.full_like(hmap1[1].dimension_values(0), np.nan))