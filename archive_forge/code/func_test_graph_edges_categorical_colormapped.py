import numpy as np
from bokeh.models import EdgesAndLinkedNodes, NodesAndLinkedEdges, NodesOnly, Patches
from bokeh.models.mappers import CategoricalColorMapper, LinearColorMapper
from holoviews.core.data import Dataset
from holoviews.element import Chord, Graph, Nodes, TriMesh, VLine, circular_layout
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.util.transform import dim
from .test_plot import TestBokehPlot, bokeh_renderer
def test_graph_edges_categorical_colormapped(self):
    g = self.graph3.opts(edge_color_index='start', edge_cmap=['#FFFFFF', '#000000'])
    plot = bokeh_renderer.get_plot(g)
    cmapper = plot.handles['edge_colormapper']
    edge_source = plot.handles['multi_line_1_source']
    glyph = plot.handles['multi_line_1_glyph']
    self.assertIsInstance(cmapper, CategoricalColorMapper)
    factors = ['0', '1', '2', '3', '4', '5', '6', '7']
    self.assertEqual(cmapper.factors, factors)
    self.assertEqual(edge_source.data['start_str__'], factors)
    self.assertEqual(property_to_dict(glyph.line_color), {'field': 'start_str__', 'transform': cmapper})