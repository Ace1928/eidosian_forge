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
def test_graph_op_edge_linewidth(self):
    edges = [(0, 1, 2), (0, 2, 10), (1, 3, 6)]
    graph = Graph(edges, vdims='line_width').opts(edge_linewidth='line_width')
    plot = mpl_renderer.get_plot(graph)
    edges = plot.handles['edges']
    self.assertEqual(edges.get_linewidths(), [2, 10, 6])