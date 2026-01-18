from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews.core.data import Dataset
from holoviews.element.chart import Points
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.graphs import Chord, Graph, Nodes, TriMesh
from holoviews.element.sankey import Sankey
from holoviews.element.util import circular_layout, connect_edges, connect_edges_pd
def test_graph_edge_segments_pd(self):
    segments = connect_edges_pd(self.graph)
    paths = []
    nodes = np.column_stack(self.nodes)
    for start, end in zip(nodes[self.source], nodes[self.target]):
        paths.append(np.array([start[:2], end[:2]]))
    self.assertEqual(segments, paths)