import numpy as np
from bokeh.models import EdgesAndLinkedNodes, NodesAndLinkedEdges, NodesOnly, Patches
from bokeh.models.mappers import CategoricalColorMapper, LinearColorMapper
from holoviews.core.data import Dataset
from holoviews.element import Chord, Graph, Nodes, TriMesh, VLine, circular_layout
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.util.transform import dim
from .test_plot import TestBokehPlot, bokeh_renderer
def test_graph_op_node_color(self):
    edges = [(0, 1), (0, 2)]
    nodes = Nodes([(0, 0, 0, 'red'), (0, 1, 1, 'green'), (1, 1, 2, 'blue')], vdims='color')
    graph = Graph((edges, nodes)).opts(node_color='color')
    plot = bokeh_renderer.get_plot(graph)
    cds = plot.handles['scatter_1_source']
    glyph = plot.handles['scatter_1_glyph']
    self.assertEqual(property_to_dict(glyph.fill_color), {'field': 'node_color'})
    self.assertEqual(glyph.line_color, 'black')
    self.assertEqual(cds.data['node_color'], np.array(['red', 'green', 'blue']))