import sys
from collections import OrderedDict
from distutils.util import strtobool
from aiokeydb.v1.exceptions import ResponseError
from aiokeydb.v1.commands.graph.edge import Edge
from aiokeydb.v1.commands.graph.exceptions import VersionMismatchException
from aiokeydb.v1.commands.graph.node import Node
from aiokeydb.v1.commands.graph.path import Path
def parse_results(self, raw_result_set):
    """
        Parse the query execution result returned from the server.
        """
    self.header = self.parse_header(raw_result_set)
    if len(self.header) == 0:
        return
    self.result_set = self.parse_records(raw_result_set)