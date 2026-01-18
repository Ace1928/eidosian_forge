import itertools
from collections import defaultdict, deque
import networkx as nx
from networkx.utils import arbitrary_element, py_random_state
def iter_neighbors(self):
    adj_node = self.adj_list
    while adj_node is not None:
        yield adj_node
        adj_node = adj_node.next