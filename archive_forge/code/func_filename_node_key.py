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
def filename_node_key(layout_node):
    """
  Return the sort key for sortable arguments nodes. This is the
  case-insensitive spelling of the first token in the node.
  """
    return layout_node.pnode.children[0].spelling.lower()