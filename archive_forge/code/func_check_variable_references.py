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
def check_variable_references(self, tree):
    """
    Check if any variable references are a case-insensitive match to any
    builtin variable names. This is probably a spelling error.
    """
    for token in tree.get_tokens(kind='semantic'):
        if token.type not in (TokenType.QUOTED_LITERAL, TokenType.DEREF):
            continue
        for varname in re.findall('\\$\\{([\\w_]+)\\}', token.spelling):
            self.check_varname(varname, token, 'Reference to')