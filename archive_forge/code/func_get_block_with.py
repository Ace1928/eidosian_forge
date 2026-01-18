from __future__ import print_function
from __future__ import unicode_literals
import collections
import logging
from cmakelang.common import InternalError
from cmakelang import lex
from cmakelang.parse.util import (
from cmakelang.parse.common import (
from cmakelang.parse.simple_nodes import (
from cmakelang.parse.statement_node import (
def get_block_with(self, node):
    """Return the block which contains the given node"""
    for block in self.blocks:
        if node in block:
            return block
    return None