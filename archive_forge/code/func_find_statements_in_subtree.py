import enum
import logging
import re
from cmakelang.common import InternalError
from cmakelang.format.formatter import get_comment_lines
from cmakelang.lex import TokenType, Token
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.body_nodes import BodyNode, FlowControlNode
from cmakelang.parse.simple_nodes import CommentNode
from cmakelang.parse.common import NodeType, TreeNode
from cmakelang.parse.statement_node import StatementNode
from cmakelang.parse.util import get_min_npargs
from cmakelang.parse import variables
from cmakelang.parse.funs.set import SetFnNode
def find_statements_in_subtree(subtree, funnames):
    """
  Return a generator that yields all statements in the `subtree` which match
  the provided set of `funnames`.
  """
    if isinstance(subtree, (list, tuple)):
        queue = subtree
    else:
        queue = [subtree]
    while queue:
        node = queue.pop(0)
        if isinstance(node, StatementNode):
            if node.get_funname() in funnames:
                yield node
        for child in node.children:
            if isinstance(child, TreeNode):
                queue.append(child)