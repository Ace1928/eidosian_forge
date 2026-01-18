import numpy as np
from bokeh.models import EdgesAndLinkedNodes, NodesAndLinkedEdges, NodesOnly, Patches
from bokeh.models.mappers import CategoricalColorMapper, LinearColorMapper
from holoviews.core.data import Dataset
from holoviews.element import Chord, Graph, Nodes, TriMesh, VLine, circular_layout
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.util.transform import dim
from .test_plot import TestBokehPlot, bokeh_renderer
def test_graph_op_edge_color_colorbar(self):
    edges = [(0, 1, 2), (0, 2, 0.5), (1, 3, 3)]
    graph = Graph(edges, vdims='color').opts(edge_color='color', colorbar=True)
    plot = bokeh_renderer.get_plot(graph)
    assert 'edge_color_colorbar' in plot.handles
    assert plot.handles['edge_color_colorbar'].color_mapper is plot.handles['edge_color_color_mapper']