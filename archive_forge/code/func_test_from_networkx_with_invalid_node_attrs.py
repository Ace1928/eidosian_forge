from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews.core.data import Dataset
from holoviews.element.chart import Points
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.graphs import Chord, Graph, Nodes, TriMesh
from holoviews.element.sankey import Sankey
from holoviews.element.util import circular_layout, connect_edges, connect_edges_pd
def test_from_networkx_with_invalid_node_attrs(self):
    import networkx as nx
    FG = nx.Graph()
    FG.add_node(1, test=[])
    FG.add_node(2, test=[])
    FG.add_edge(1, 2)
    graph = Graph.from_networkx(FG, nx.circular_layout)
    self.assertEqual(graph.nodes.vdims, [])
    self.assertEqual(graph.nodes.dimension_values(2), np.array([1, 2]))
    self.assertEqual(graph.array(), np.array([(1, 2)]))