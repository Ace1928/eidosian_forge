from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews.core.data import Dataset
from holoviews.element.chart import Points
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.graphs import Chord, Graph, Nodes, TriMesh
from holoviews.element.sankey import Sankey
from holoviews.element.util import circular_layout, connect_edges, connect_edges_pd
def test_chord_constructor_self_reference(self):
    chord = Chord([('A', 'B', 2), ('B', 'A', 3), ('A', 'A', 2)])
    nodes = pd.DataFrame([[-0.5, 0.866025, 'A'], [0.5, -0.866025, 'B']], columns=['x', 'y', 'index'])
    self.assertEqual(chord.nodes, Nodes(nodes))