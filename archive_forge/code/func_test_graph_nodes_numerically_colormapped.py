import numpy as np
from bokeh.models import EdgesAndLinkedNodes, NodesAndLinkedEdges, NodesOnly, Patches
from bokeh.models.mappers import CategoricalColorMapper, LinearColorMapper
from holoviews.core.data import Dataset
from holoviews.element import Chord, Graph, Nodes, TriMesh, VLine, circular_layout
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.util.transform import dim
from .test_plot import TestBokehPlot, bokeh_renderer
def test_graph_nodes_numerically_colormapped(self):
    g = self.graph3.opts(color_index='Weight', cmap='viridis')
    plot = bokeh_renderer.get_plot(g)
    cmapper = plot.handles['color_mapper']
    node_source = plot.handles['scatter_1_source']
    glyph = plot.handles['scatter_1_glyph']
    self.assertIsInstance(cmapper, LinearColorMapper)
    self.assertEqual(cmapper.low, self.weights.min())
    self.assertEqual(cmapper.high, self.weights.max())
    self.assertEqual(node_source.data['Weight'], self.node_info2['Weight'])
    self.assertEqual(property_to_dict(glyph.fill_color), {'field': 'Weight', 'transform': cmapper})