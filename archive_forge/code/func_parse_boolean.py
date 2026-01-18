import sys
from collections import OrderedDict
from distutils.util import strtobool
from aiokeydb.v1.exceptions import ResponseError
from aiokeydb.v1.commands.graph.edge import Edge
from aiokeydb.v1.commands.graph.exceptions import VersionMismatchException
from aiokeydb.v1.commands.graph.node import Node
from aiokeydb.v1.commands.graph.path import Path
def parse_boolean(self, value):
    """
        Parse the cell value as a boolean.
        """
    value = value.decode() if isinstance(value, bytes) else value
    try:
        scalar = strtobool(value)
    except ValueError:
        sys.stderr.write('unknown boolean type\n')
        scalar = None
    return scalar