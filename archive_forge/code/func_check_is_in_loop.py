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
def check_is_in_loop(self, node):
    """Ensure that a break() or continue() statement has a foreach() or
      while() node in it's ancestry."""
    _, local_ctx = self.context
    prevparent = node
    parent = node.parent
    while parent:
        if not isinstance(parent, FlowControlNode):
            prevparent = parent
            parent = parent.parent
            continue
        block = parent.get_block_with(prevparent)
        if block is None:
            prevparent = parent
            parent = parent.parent
            continue
        if block.open_stmt.get_funname() in ('foreach', 'while'):
            return True
        prevparent = parent
        parent = parent.parent
    local_ctx.record_lint('E0103', node.get_funname(), location=node.get_location())