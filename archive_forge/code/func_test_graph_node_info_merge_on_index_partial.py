from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews.core.data import Dataset
from holoviews.element.chart import Points
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.graphs import Chord, Graph, Nodes, TriMesh
from holoviews.element.sankey import Sankey
from holoviews.element.util import circular_layout, connect_edges, connect_edges_pd
def test_graph_node_info_merge_on_index_partial(self):
    node_info = Dataset((np.arange(5), np.arange(1, 6)), 'index', 'label')
    graph = Graph(((self.source, self.target), node_info))
    expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, np.nan, np.nan, np.nan])
    self.assertEqual(graph.nodes.dimension_values(3), expected)