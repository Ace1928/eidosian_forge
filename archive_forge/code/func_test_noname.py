import os
import tempfile
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_noname(self):
    line = '*network\n'
    other_lines = self.data.split('\n')[1:]
    data = line + '\n'.join(other_lines)
    G = nx.parse_pajek(data)