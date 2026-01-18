from aiokeydb.v1.commands.graph.edge import Edge
from aiokeydb.v1.commands.graph.node import Node
def get_relationship(self, index):
    return self._edges[index]