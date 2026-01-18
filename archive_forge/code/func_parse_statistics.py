import sys
from collections import OrderedDict
from distutils.util import strtobool
from aiokeydb.v1.exceptions import ResponseError
from aiokeydb.v1.commands.graph.edge import Edge
from aiokeydb.v1.commands.graph.exceptions import VersionMismatchException
from aiokeydb.v1.commands.graph.node import Node
from aiokeydb.v1.commands.graph.path import Path
def parse_statistics(self, raw_statistics):
    """
        Parse the statistics returned in the response.
        """
    self.statistics = {}
    for idx, stat in enumerate(raw_statistics):
        if isinstance(stat, bytes):
            raw_statistics[idx] = stat.decode()
    for s in STATS:
        v = self._get_value(s, raw_statistics)
        if v is not None:
            self.statistics[s] = v