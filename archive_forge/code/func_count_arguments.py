from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import io
import logging
import re
import sys
from cmakelang import lex
from cmakelang import markup
from cmakelang.common import UserError
from cmakelang.lex import TokenType
from cmakelang.parse.argument_nodes import PositionalGroupNode
from cmakelang.parse.common import FlowType, NodeType, TreeNode
from cmakelang.parse.util import comment_is_tag
from cmakelang.parse import simple_nodes
def count_arguments(children):
    """
  Count the number of positional arguments (excluding line comments and
  whitespace) within a parg group.
  """
    count = 0
    for child in children:
        if child.node_type is NodeType.COMMENT:
            continue
        count += 1
    return count