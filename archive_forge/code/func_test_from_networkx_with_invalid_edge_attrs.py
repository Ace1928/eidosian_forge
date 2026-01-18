from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews.core.data import Dataset
from holoviews.element.chart import Points
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.graphs import Chord, Graph, Nodes, TriMesh
from holoviews.element.sankey import Sankey
from holoviews.element.util import circular_layout, connect_edges, connect_edges_pd
def test_from_networkx_with_invalid_edge_attrs(self):
    import networkx as nx
    FG = nx.Graph()
    FG.add_weighted_edges_from([(1, 2, []), (1, 3, []), (2, 4, []), (3, 4, [])])
    graph = Graph.from_networkx(FG, nx.circular_layout)
    self.assertEqual(graph.vdims, [])