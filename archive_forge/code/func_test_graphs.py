import pytest
import networkx as nx
from networkx.utils import pairwise
def test_graphs(self):
    validate_path(self.XG, 's', 'v', 9, nx.johnson(self.XG)['s']['v'])
    validate_path(self.MXG, 's', 'v', 9, nx.johnson(self.MXG)['s']['v'])
    validate_path(self.XG2, 1, 3, 4, nx.johnson(self.XG2)[1][3])
    validate_path(self.XG3, 0, 3, 15, nx.johnson(self.XG3)[0][3])
    validate_path(self.XG4, 0, 2, 4, nx.johnson(self.XG4)[0][2])
    validate_path(self.MXG4, 0, 2, 4, nx.johnson(self.MXG4)[0][2])