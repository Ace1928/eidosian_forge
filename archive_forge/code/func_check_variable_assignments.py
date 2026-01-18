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
def check_variable_assignments(self, tree):
    """
    Check if any variable assignments are a case-insensitive match to any
    builtin variable names. This is probably a spelling error.
    """
    for stmt in find_statements_in_subtree(tree, ['set', 'list']):
        if stmt.get_funname() == 'set':
            token = stmt.argtree.varname
        elif stmt.get_funname() == 'list':
            token = stmt.argtree.parg_groups[0].get_tokens(kind='semantic')[1]
        else:
            continue
        self.check_varname(token.spelling, token, 'Assignment to')