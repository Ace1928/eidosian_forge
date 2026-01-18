import numpy as np
from bokeh.models import EdgesAndLinkedNodes, NodesAndLinkedEdges, NodesOnly, Patches
from bokeh.models.mappers import CategoricalColorMapper, LinearColorMapper
from holoviews.core.data import Dataset
from holoviews.element import Chord, Graph, Nodes, TriMesh, VLine, circular_layout
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.util.transform import dim
from .test_plot import TestBokehPlot, bokeh_renderer
def test_graph_op_edge_color(self):
    edges = [(0, 1, 'red'), (0, 2, 'green'), (1, 3, 'blue')]
    graph = Graph(edges, vdims='color').opts(edge_color='color')
    plot = bokeh_renderer.get_plot(graph)
    cds = plot.handles['multi_line_1_source']
    glyph = plot.handles['multi_line_1_glyph']
    self.assertEqual(property_to_dict(glyph.line_color), {'field': 'edge_color'})
    self.assertEqual(cds.data['edge_color'], np.array(['red', 'green', 'blue']))