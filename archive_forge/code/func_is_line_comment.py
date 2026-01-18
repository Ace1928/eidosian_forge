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
def is_line_comment(node):
    """
  Return true if the node is a pure parser node holding a line comment (i.e.
  not a bracket comment)
  """
    if isinstance(node, CommentNode):
        node = node.pnode
    if not isinstance(node, TreeNode):
        return False
    if not node.children:
        return False
    return node.children[-1].type == TokenType.COMMENT