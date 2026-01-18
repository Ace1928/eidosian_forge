import numpy as np
from bokeh.models import EdgesAndLinkedNodes, NodesAndLinkedEdges, NodesOnly, Patches
from bokeh.models.mappers import CategoricalColorMapper, LinearColorMapper
from holoviews.core.data import Dataset
from holoviews.element import Chord, Graph, Nodes, TriMesh, VLine, circular_layout
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.util.transform import dim
from .test_plot import TestBokehPlot, bokeh_renderer
def test_graph_nodes_categorical_colormapped(self):
    g = self.graph2.opts(color_index='Label', cmap='Set1')
    plot = bokeh_renderer.get_plot(g)
    cmapper = plot.handles['color_mapper']
    node_source = plot.handles['scatter_1_source']
    glyph = plot.handles['scatter_1_glyph']
    self.assertIsInstance(cmapper, CategoricalColorMapper)
    self.assertEqual(cmapper.factors, ['Output', 'Input'])
    self.assertEqual(node_source.data['Label'], self.node_info['Label'])
    self.assertEqual(property_to_dict(glyph.fill_color), {'field': 'Label', 'transform': cmapper})