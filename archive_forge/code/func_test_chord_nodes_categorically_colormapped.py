import numpy as np
from bokeh.models import EdgesAndLinkedNodes, NodesAndLinkedEdges, NodesOnly, Patches
from bokeh.models.mappers import CategoricalColorMapper, LinearColorMapper
from holoviews.core.data import Dataset
from holoviews.element import Chord, Graph, Nodes, TriMesh, VLine, circular_layout
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.util.transform import dim
from .test_plot import TestBokehPlot, bokeh_renderer
def test_chord_nodes_categorically_colormapped(self):
    g = self.chord.opts(color_index='Label', label_index='Label', cmap=['#FFFFFF', '#888888', '#000000'])
    plot = bokeh_renderer.get_plot(g)
    cmapper = plot.handles['color_mapper']
    source = plot.handles['scatter_1_source']
    arc_source = plot.handles['multi_line_2_source']
    glyph = plot.handles['scatter_1_glyph']
    self.assertIsInstance(cmapper, CategoricalColorMapper)
    self.assertEqual(cmapper.factors, ['A', 'B', 'C'])
    self.assertEqual(cmapper.palette, ['#FFFFFF', '#888888', '#000000'])
    self.assertEqual(source.data['Label'], np.array(['A', 'B', 'C']))
    self.assertEqual(arc_source.data['Label'], np.array(['A', 'B', 'C']))
    self.assertEqual(property_to_dict(glyph.fill_color), {'field': 'Label', 'transform': cmapper})