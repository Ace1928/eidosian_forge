import os
import six
import sys
from pyparsing import (alphanums, Empty, Group, locatedExpr,
from . import console
from . import log
from . import prefs
from .node import ConfigNode, ExecutionError
import signal
def attach_root_node(self, root_node):
    """
        @param root_node: The root ConfigNode object
        @type root_node: ConfigNode
        """
    self._current_node = root_node
    self._root_node = root_node