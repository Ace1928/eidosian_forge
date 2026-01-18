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
def test_graph_op_edge_color_update(self):
    graph = HoloMap({0: Graph([(0, 1, 'red'), (0, 2, 'green'), (1, 3, 'blue')], vdims='color'), 1: Graph([(0, 1, 'green'), (0, 2, 'blue'), (1, 3, 'red')], vdims='color')}).opts(edge_color='color')
    plot = mpl_renderer.get_plot(graph)
    edges = plot.handles['edges']
    self.assertEqual(edges.get_edgecolors(), np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 0.50196078, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]]))
    plot.update((1,))
    self.assertEqual(edges.get_edgecolors(), np.array([[0.0, 0.50196078, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0]]))