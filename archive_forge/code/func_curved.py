import collections
import itertools
from numbers import Number
import networkx as nx
from networkx.drawing.layout import (
def curved(self, edge_index):
    return self.base_connection_styles[edge_index % self.n]