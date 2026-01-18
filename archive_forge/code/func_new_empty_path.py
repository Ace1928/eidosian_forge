from aiokeydb.v1.commands.graph.edge import Edge
from aiokeydb.v1.commands.graph.node import Node
@classmethod
def new_empty_path(cls):
    return cls([], [])