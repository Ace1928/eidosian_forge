import numpy as np
from bokeh.models import EdgesAndLinkedNodes, NodesAndLinkedEdges, NodesOnly, Patches
from bokeh.models.mappers import CategoricalColorMapper, LinearColorMapper
from holoviews.core.data import Dataset
from holoviews.element import Chord, Graph, Nodes, TriMesh, VLine, circular_layout
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.util.transform import dim
from .test_plot import TestBokehPlot, bokeh_renderer
def test_graph_op_edge_line_width(self):
    edges = [(0, 1, 2), (0, 2, 10), (1, 3, 6)]
    graph = Graph(edges, vdims='line_width').opts(edge_line_width='line_width')
    plot = bokeh_renderer.get_plot(graph)
    cds = plot.handles['multi_line_1_source']
    glyph = plot.handles['multi_line_1_glyph']
    self.assertEqual(property_to_dict(glyph.line_width), {'field': 'edge_line_width'})
    self.assertEqual(cds.data['edge_line_width'], np.array([2, 10, 6]))