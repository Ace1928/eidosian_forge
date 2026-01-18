import networkx as nx
from collections import deque
def indegree(self, vertex):
    return len([e for e in self.incidence_dict[vertex] if e.head is vertex])