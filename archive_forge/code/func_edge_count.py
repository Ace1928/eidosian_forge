from aiokeydb.v1.commands.graph.edge import Edge
from aiokeydb.v1.commands.graph.node import Node
def edge_count(self):
    return len(self._edges)