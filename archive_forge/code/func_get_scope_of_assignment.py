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
def get_scope_of_assignment(set_node):
    """Return the scope of the assignment."""
    args = set_node.argtree
    if isinstance(args, SetFnNode):
        if args.cache:
            if args.cache.type == 'INTERNAL':
                return Scope.INTERNAL
            return Scope.CACHE
        if args.parent_scope:
            return Scope.PARENT
    prev = set_node
    parent = args.parent
    while parent:
        if isinstance(parent, FlowControlNode):
            block = parent.get_block_with(prev)
            if block.open_stmt.get_funname() == 'function':
                return Scope.LOCAL
            if block.open_stmt.get_funname() == 'macro':
                return Scope.PARENT
        prev = parent
        parent = parent.parent
    return Scope.DIRECTORY