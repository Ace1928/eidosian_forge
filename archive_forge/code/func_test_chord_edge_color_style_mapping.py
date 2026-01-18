import numpy as np
from bokeh.models import EdgesAndLinkedNodes, NodesAndLinkedEdges, NodesOnly, Patches
from bokeh.models.mappers import CategoricalColorMapper, LinearColorMapper
from holoviews.core.data import Dataset
from holoviews.element import Chord, Graph, Nodes, TriMesh, VLine, circular_layout
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.util.transform import dim
from .test_plot import TestBokehPlot, bokeh_renderer
def test_chord_edge_color_style_mapping(self):
    g = self.chord.opts(edge_color=dim('start').astype(str), edge_cmap=['#FFFFFF', '#000000'])
    plot = bokeh_renderer.get_plot(g)
    cmapper = plot.handles['edge_color_color_mapper']
    edge_source = plot.handles['multi_line_1_source']
    glyph = plot.handles['multi_line_1_glyph']
    self.assertIsInstance(cmapper, CategoricalColorMapper)
    self.assertEqual(cmapper.palette, ['#FFFFFF', '#000000', '#FFFFFF'])
    self.assertEqual(cmapper.factors, ['0', '1', '2'])
    self.assertEqual(edge_source.data['edge_color'], np.array(['0', '0', '1']))
    self.assertEqual(property_to_dict(glyph.line_color), {'field': 'edge_color', 'transform': cmapper})