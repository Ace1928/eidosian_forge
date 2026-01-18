import numpy as np
from bokeh.models import EdgesAndLinkedNodes, NodesAndLinkedEdges, NodesOnly, Patches
from bokeh.models.mappers import CategoricalColorMapper, LinearColorMapper
from holoviews.core.data import Dataset
from holoviews.element import Chord, Graph, Nodes, TriMesh, VLine, circular_layout
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.util.transform import dim
from .test_plot import TestBokehPlot, bokeh_renderer
def test_graph_edges_numerically_colormapped(self):
    g = self.graph4.opts(edge_color_index='Weight', edge_cmap=['#FFFFFF', '#000000'])
    plot = bokeh_renderer.get_plot(g)
    cmapper = plot.handles['edge_colormapper']
    edge_source = plot.handles['multi_line_1_source']
    glyph = plot.handles['multi_line_1_glyph']
    self.assertIsInstance(cmapper, LinearColorMapper)
    self.assertEqual(cmapper.low, self.weights.min())
    self.assertEqual(cmapper.high, self.weights.max())
    self.assertEqual(edge_source.data['Weight'], self.node_info2['Weight'])
    self.assertEqual(property_to_dict(glyph.line_color), {'field': 'Weight', 'transform': cmapper})