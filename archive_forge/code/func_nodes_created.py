import sys
from collections import OrderedDict
from distutils.util import strtobool
from aiokeydb.v1.exceptions import ResponseError
from aiokeydb.v1.commands.graph.edge import Edge
from aiokeydb.v1.commands.graph.exceptions import VersionMismatchException
from aiokeydb.v1.commands.graph.node import Node
from aiokeydb.v1.commands.graph.path import Path
@property
def nodes_created(self):
    """Returns the number of nodes created in the query"""
    return self._get_stat(NODES_CREATED)