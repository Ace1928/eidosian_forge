from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews.core.data import Dataset
from holoviews.element.chart import Points
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.graphs import Chord, Graph, Nodes, TriMesh
from holoviews.element.sankey import Sankey
from holoviews.element.util import circular_layout, connect_edges, connect_edges_pd
def test_from_networkx_only_nodes(self):
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3])
    graph = Graph.from_networkx(G, nx.circular_layout)
    self.assertEqual(graph.nodes.dimension_values(2), np.array([1, 2, 3]))