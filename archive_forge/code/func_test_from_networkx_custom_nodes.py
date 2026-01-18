from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews.core.data import Dataset
from holoviews.element.chart import Points
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.graphs import Chord, Graph, Nodes, TriMesh
from holoviews.element.sankey import Sankey
from holoviews.element.util import circular_layout, connect_edges, connect_edges_pd
def test_from_networkx_custom_nodes(self):
    import networkx as nx
    FG = nx.Graph()
    FG.add_weighted_edges_from([(1, 2, 0.125), (1, 3, 0.75), (2, 4, 1.2), (3, 4, 0.375)])
    nodes = Dataset([(1, 'A'), (2, 'B'), (3, 'A'), (4, 'B')], 'index', 'some_attribute')
    graph = Graph.from_networkx(FG, nx.circular_layout, nodes=nodes)
    self.assertEqual(graph.nodes.dimension_values('some_attribute'), np.array(['A', 'B', 'A', 'B']))