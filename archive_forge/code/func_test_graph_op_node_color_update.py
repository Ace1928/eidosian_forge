import numpy as np
import pytest
from matplotlib.collections import LineCollection, PolyCollection
from packaging.version import Version
from holoviews.core.data import Dataset
from holoviews.core.options import AbbreviatedException, Cycle
from holoviews.core.spaces import HoloMap
from holoviews.element import Chord, Graph, Nodes, TriMesh, circular_layout
from holoviews.util.transform import dim
from .test_plot import TestMPLPlot, mpl_renderer
def test_graph_op_node_color_update(self):
    edges = [(0, 1), (0, 2)]

    def get_graph(i):
        c1, c2, c3 = {0: ('#00FF00', '#0000FF', '#FF0000'), 1: ('#FF0000', '#00FF00', '#0000FF')}[i]
        nodes = Nodes([(0, 0, 0, c1), (0, 1, 1, c2), (1, 1, 2, c3)], vdims='color')
        return Graph((edges, nodes))
    graph = HoloMap({0: get_graph(0), 1: get_graph(1)}).opts(node_color='color')
    plot = mpl_renderer.get_plot(graph)
    artist = plot.handles['nodes']
    self.assertEqual(artist.get_facecolors(), np.array([[0, 1, 0, 1], [0, 0, 1, 1], [1, 0, 0, 1]]))
    plot.update((1,))
    self.assertEqual(artist.get_facecolors(), np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]))