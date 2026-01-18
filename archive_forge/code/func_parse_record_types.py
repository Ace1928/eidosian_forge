import sys
from collections import OrderedDict
from distutils.util import strtobool
from aiokeydb.v1.exceptions import ResponseError
from aiokeydb.v1.commands.graph.edge import Edge
from aiokeydb.v1.commands.graph.exceptions import VersionMismatchException
from aiokeydb.v1.commands.graph.node import Node
from aiokeydb.v1.commands.graph.path import Path
@property
def parse_record_types(self):
    return {ResultSetColumnTypes.COLUMN_SCALAR: self.parse_scalar, ResultSetColumnTypes.COLUMN_NODE: self.parse_node, ResultSetColumnTypes.COLUMN_RELATION: self.parse_edge, ResultSetColumnTypes.COLUMN_UNKNOWN: self.parse_unknown}