import sys
from collections import OrderedDict
from distutils.util import strtobool
from aiokeydb.v1.exceptions import ResponseError
from aiokeydb.v1.commands.graph.edge import Edge
from aiokeydb.v1.commands.graph.exceptions import VersionMismatchException
from aiokeydb.v1.commands.graph.node import Node
from aiokeydb.v1.commands.graph.path import Path
def parse_edge(self, cell):
    """
        Parse the cell to an edge.
        """
    edge_id = int(cell[0])
    relation = self.graph.get_relation(cell[1])
    src_node_id = int(cell[2])
    dest_node_id = int(cell[3])
    properties = self.parse_entity_properties(cell[4])
    return Edge(src_node_id, relation, dest_node_id, edge_id=edge_id, properties=properties)