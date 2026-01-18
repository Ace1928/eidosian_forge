import sys
from collections import OrderedDict
from distutils.util import strtobool
from aiokeydb.v1.exceptions import ResponseError
from aiokeydb.v1.commands.graph.edge import Edge
from aiokeydb.v1.commands.graph.exceptions import VersionMismatchException
from aiokeydb.v1.commands.graph.node import Node
from aiokeydb.v1.commands.graph.path import Path
def parse_point(self, cell):
    """
        Parse the cell to point.
        """
    p = {}
    p['latitude'] = float(cell[0])
    p['longitude'] = float(cell[1])
    return p