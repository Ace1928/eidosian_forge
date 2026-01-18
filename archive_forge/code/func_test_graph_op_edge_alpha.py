import numpy as np
from bokeh.models import EdgesAndLinkedNodes, NodesAndLinkedEdges, NodesOnly, Patches
from bokeh.models.mappers import CategoricalColorMapper, LinearColorMapper
from holoviews.core.data import Dataset
from holoviews.element import Chord, Graph, Nodes, TriMesh, VLine, circular_layout
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.util.transform import dim
from .test_plot import TestBokehPlot, bokeh_renderer
def test_graph_op_edge_alpha(self):
    edges = [(0, 1, 0.1), (0, 2, 0.5), (1, 3, 0.3)]
    graph = Graph(edges, vdims='alpha').opts(edge_alpha='alpha')
    plot = bokeh_renderer.get_plot(graph)
    cds = plot.handles['multi_line_1_source']
    glyph = plot.handles['multi_line_1_glyph']
    self.assertEqual(property_to_dict(glyph.line_alpha), {'field': 'edge_alpha'})
    self.assertEqual(cds.data['edge_alpha'], np.array([0.1, 0.5, 0.3]))