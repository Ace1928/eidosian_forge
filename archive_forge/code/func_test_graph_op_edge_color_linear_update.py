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
def test_graph_op_edge_color_linear_update(self):
    graph = HoloMap({0: Graph([(0, 1, 2), (0, 2, 0.5), (1, 3, 3)], vdims='color'), 1: Graph([(0, 1, 4.3), (0, 2, 1.4), (1, 3, 2.6)], vdims='color')}).opts(edge_color='color', framewise=True)
    plot = mpl_renderer.get_plot(graph)
    edges = plot.handles['edges']
    self.assertEqual(np.asarray(edges.get_array()), np.array([2, 0.5, 3]))
    self.assertEqual(edges.get_clim(), (0.5, 3))
    plot.update((1,))
    self.assertEqual(np.asarray(edges.get_array()), np.array([4.3, 1.4, 2.6]))
    self.assertEqual(edges.get_clim(), (1.4, 4.3))